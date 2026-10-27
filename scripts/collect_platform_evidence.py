"""Collect platform-level execution evidence as text: UC table inventory and lineage,
model registry state, Lakebase serving table, Databricks App status, Genie Agent.

Usage:
  python -I scripts/collect_platform_evidence.py --profile fevm-classic-stable --since 2026-10-27
"""
import argparse
import json
import subprocess
from datetime import datetime, timezone

CATALOG, SCHEMA = "bhavik_demos", "offer_prioritization"
MODEL = f"{CATALOG}.{SCHEMA}.healthcare_offer_prioritizer"
APP = "offer-prioritization-console"


def cli(profile, *args):
    out = subprocess.run(["databricks", *args, "--profile", profile, "-o", "json"],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out) if out.strip() else {}


def sql(profile, statement):
    out = subprocess.run(["databricks", "experimental", "aitools", "tools", "query", statement,
                          "--profile", profile], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    return "\n".join(lines + ["| " + " | ".join(str(r.get(c, "")).replace("|", "/") for c in cols) + " |"
                              for r in rows])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--since", required=True, help="only lineage events on/after this date")
    ap.add_argument("--genie-space-id", required=True)
    ap.add_argument("--out", default="evidence/platform.md")
    a = ap.parse_args()
    p = a.profile
    md = [f"# Platform evidence: {CATALOG}.{SCHEMA}", "",
          f"Collected {datetime.now(timezone.utc).isoformat(timespec='seconds')} from workspace "
          f"`fevm-classic-stable` with the Databricks CLI (`scripts/collect_platform_evidence.py`).", ""]

    tables = sql(p, f"""
        SELECT table_name, table_type, comment FROM {CATALOG}.information_schema.tables
        WHERE table_schema = '{SCHEMA}' AND table_name NOT LIKE 'event_log_%' AND table_name NOT LIKE '\\_\\_%'
        ORDER BY table_name""")
    for t in tables:
        t["row_count"] = sql(p, f"SELECT count(*) AS n FROM {CATALOG}.{SCHEMA}.{t['table_name']}")[0]["n"]
        t["comment"] = (t.get("comment") or "")[:110]
    md += ["## 1. Unity Catalog: governed assets", "",
           "Bronze and silver streaming tables accumulate every landed extract batch (Auto Loader is incremental:",
           "each job run lands one new batch). Gold views, features and the model use the latest `batch_id` only.", "",
           md_table(tables, ["table_name", "table_type", "row_count", "comment"]), ""]

    tags = sql(p, f"""
        SELECT 'table' AS level, table_name AS object, tag_name, tag_value FROM {CATALOG}.information_schema.table_tags
        WHERE schema_name = '{SCHEMA}'
        UNION ALL
        SELECT 'column', concat(table_name, '.', column_name), tag_name, tag_value FROM {CATALOG}.information_schema.column_tags
        WHERE schema_name = '{SCHEMA}' ORDER BY 1 DESC, 2, 3""")
    md += ["### Governed tags", "", md_table(tags, ["level", "object", "tag_name", "tag_value"]), ""]

    lineage = sql(p, f"""
        SELECT coalesce(source_table_name, '(external / file)') AS source,
               coalesce(target_table_name, '(read by job)') AS target,
               entity_type, count(*) AS events, max(event_time) AS last_seen
        FROM system.access.table_lineage
        WHERE event_time >= '{a.since}'
          AND ((target_table_catalog = '{CATALOG}' AND target_table_schema = '{SCHEMA}')
            OR (source_table_catalog = '{CATALOG}' AND source_table_schema = '{SCHEMA}'))
          AND coalesce(target_table_name, '') NOT LIKE 'event_log_%' AND coalesce(source_table_name, '') NOT LIKE 'event_log_%'
          AND coalesce(target_table_name, '') NOT LIKE '\\_probe'
        GROUP BY ALL ORDER BY entity_type DESC, source, target""")
    md += ["### Lineage (system.access.table_lineage)", "",
           "Pipeline hops are table-to-table. The ML notebooks read silver/features into pandas, so UC records those",
           "steps as job reads and job writes rather than a direct edge.", "",
           md_table(lineage, ["source", "target", "entity_type", "events", "last_seen"]), ""]

    versions = cli(p, "model-versions", "list", MODEL)
    versions = versions if isinstance(versions, list) else versions.get("model_versions", [])
    champion = cli(p, "model-versions", "get-by-alias", MODEL, "champion")
    md += ["## 2. Model registry (Unity Catalog)", "",
           f"`{MODEL}`: `@champion` -> **v{champion.get('version')}** (run `{champion.get('run_id')}`)", "",
           md_table([{"version": v.get("version"), "status": v.get("status"),
                      "created": datetime.fromtimestamp(v["created_at"] / 1000, timezone.utc).isoformat(timespec="seconds"),
                      "comment": v.get("comment", "")} for v in sorted(versions, key=lambda v: int(v["version"]))],
                    ["version", "status", "created", "comment"]), ""]

    recs = sql(p, f"""
        SELECT rank, count(*) AS recommendations, round(avg(priority_score), 1) AS avg_score,
               min(priority_score) AS min_score, max(priority_score) AS max_score
        FROM {CATALOG}.{SCHEMA}.offer_recommendations GROUP BY rank ORDER BY rank""")
    sample = sql(p, f"""
        SELECT member_id, rank, offer_name, priority_score, reasoning_source, llm_reasoning
        FROM {CATALOG}.{SCHEMA}.member_offer_recommendations_with_reasoning
        WHERE member_id = (SELECT member_id FROM {CATALOG}.{SCHEMA}.member_offer_recommendations_with_reasoning
                           ORDER BY risk_score DESC LIMIT 1) ORDER BY rank""")
    md += ["## 3. Model output", "", "### offer_recommendations by rank", "",
           md_table(recs, ["rank", "recommendations", "avg_score", "min_score", "max_score"]), "",
           "### Work-queue sample: highest-risk member, with LLM reasoning (databricks-gpt-5-1)", ""]
    for r in sample:
        md += [f"**#{r['rank']} {r['offer_name']}** (score {r['priority_score']}, source: {r['reasoning_source']})",
               "", f"> {r['llm_reasoning']}", ""]

    app = cli(p, "apps", "get", APP)
    md += ["## 4. Databricks App + Lakebase", "", md_table([{
        "app": app.get("name"), "url": app.get("url"), "app_status": app.get("app_status", {}).get("state"),
        "compute": app.get("compute_status", {}).get("state"),
        "active_deployment": app.get("active_deployment", {}).get("deployment_id"),
        "deployed_at": app.get("active_deployment", {}).get("update_time"),
        "resources": ", ".join(f"{r['name']}={list(k for k in r if k != 'name')[0]}" for r in app.get("resources", [])),
    }], ["app", "url", "app_status", "compute", "active_deployment", "deployed_at", "resources"]), ""]
    endpoint = cli(p, "postgres", "list-endpoints", "projects/offer-prioritization/branches/production")
    endpoint = endpoint if isinstance(endpoint, list) else endpoint.get("endpoints", [])
    md += [md_table([{"endpoint": e.get("name"), "host": e.get("status", {}).get("hosts", {}).get("host"),
                      "state": e.get("status", {}).get("current_state")} for e in endpoint],
                    ["endpoint", "host", "state"]), ""]

    space = cli(p, "genie", "get-space", a.genie_space_id)
    md += ["## 5. Genie Agent", "", md_table([{"title": space.get("title"), "space_id": a.genie_space_id,
                                               "warehouse_id": space.get("warehouse_id"),
                                               "description": space.get("description", "")[:120]}],
                                             ["title", "space_id", "warehouse_id", "description"]), "",
           "Question-and-answer transcript with generated SQL: [notebooks/genie_agent.md](notebooks/genie_agent.md).", ""]

    with open(a.out, "w") as f:
        f.write("\n".join(md))
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
