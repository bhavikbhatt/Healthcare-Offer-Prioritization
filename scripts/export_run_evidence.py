"""Export execution evidence for a run of the offer-prioritization-e2e job as text.

For every notebook task: the executed notebook (code + real cell outputs) as
evidence/notebooks/<task>.ipynb and a readable evidence/notebooks/<task>.md.
For the Lakeflow pipeline task: the update's flow progress and data-quality
expectation metrics from the pipeline event log.
Also writes evidence/job_run.json (task states, durations, notebook exit values).

Usage:
  python -I scripts/export_run_evidence.py --profile fevm-classic-stable --run-id <job_run_id>
"""
import argparse
import base64
import html
import json
import os
import re
import subprocess
import urllib.parse
from datetime import datetime, timezone


def cli(profile, *args):
    out = subprocess.run(["databricks", *args, "--profile", profile, "-o", "json"],
                         check=True, capture_output=True, text=True).stdout
    return json.loads(out) if out.strip() else {}


def notebook_model(export):
    page = export["views"][0]["content"]
    m = re.search(r"__DATABRICKS_NOTEBOOK_MODEL = '([^']+)'", page)
    return json.loads(urllib.parse.unquote(base64.b64decode(m.group(1)).decode("utf-8")))


def strip_html(s):
    s = re.sub(r"<br\s*/?>", "\n", s)
    s = re.sub(r"<[^>]+>", "", s)
    return html.unescape(s).strip("\n")


def result_text(res):
    """Flatten a command's results into plain text (stdout, tables, errors)."""
    if not res:
        return ""
    kind, data = res.get("type"), res.get("data")
    if kind == "listResults":
        return "\n".join(t for t in (result_text(r) for r in data or []) if t)
    if kind == "table":
        cols = [c["name"] for c in res.get("schema", [])]
        rows = data or []
        lines = [" | ".join(cols), " | ".join("---" for _ in cols)]
        lines += [" | ".join("" if v is None else str(v).replace("\n", " ")[:160] for v in r) for r in rows[:50]]
        if len(rows) > 50:
            lines.append(f"... ({len(rows)} rows total)")
        return "\n".join(lines)
    if kind in ("html", "htmlSandbox"):
        return strip_html(data or "")
    if kind in ("text", "ansi", "raw"):
        return str(data or "")
    if kind == "error":
        return strip_html(str(data or res.get("summary", "")))
    if kind == "image":
        return "[image output omitted]"
    return str(data) if data else ""


ANSI = re.compile(r"\x1b\[[0-9;]*m")
NOISE = ("{'text/plain': 'Downloading artifacts", "{'text/plain': 'Uploading artifacts")


def clean(text):
    lines = [ANSI.sub("", l) for l in text.splitlines()]
    return "\n".join(l for l in lines if not l.startswith(NOISE)).strip("\n")


def to_ipynb(model):
    cells = []
    for c in sorted(model.get("commands", []), key=lambda x: x.get("position", 0)):
        src = c.get("command", "")
        if src.lstrip().startswith("%md"):
            cells.append({"cell_type": "markdown", "metadata": {},
                          "source": src.lstrip()[3:].lstrip("\n").splitlines(keepends=True)})
            continue
        text = clean(result_text(c.get("results")))
        err = c.get("error") or c.get("errorSummary")
        outputs = []
        if text:
            outputs.append({"output_type": "stream", "name": "stdout", "text": text.splitlines(keepends=True)})
        if err:
            outputs.append({"output_type": "stream", "name": "stderr",
                            "text": strip_html(str(err)).splitlines(keepends=True)})
        cells.append({"cell_type": "code", "execution_count": None, "metadata": {},
                      "source": src.splitlines(keepends=True), "outputs": outputs})
    return {"nbformat": 4, "nbformat_minor": 5, "cells": cells,
            "metadata": {"language_info": {"name": "python"}}}


def to_markdown(task_key, run, nb):
    md = [f"# Executed notebook: `{task_key}`", "",
          f"- Job run task id: `{run['run_id']}`",
          f"- Result: **{run['state'].get('result_state')}**",
          f"- Started: {datetime.fromtimestamp(run['start_time']/1000, timezone.utc).isoformat()}",
          f"- Duration: {round((run.get('end_time', 0) - run['start_time'])/1000)} s", ""]
    for cell in nb["cells"]:
        if cell["cell_type"] == "markdown":
            md += ["".join(cell["source"]), ""]
            continue
        md += ["```python", "".join(cell["source"]).rstrip(), "```"]
        for o in cell["outputs"]:
            md += ["Output:" if o["name"] == "stdout" else "Error:", "```text",
                   "".join(o["text"]).rstrip()[:20000], "```"]
        md.append("")
    return "\n".join(md)


def sql(profile, statement):
    out = subprocess.run(["databricks", "experimental", "aitools", "tools", "query", statement,
                          "--profile", profile], check=True, capture_output=True, text=True).stdout
    return json.loads(out)


def md_table(rows, cols):
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    return "\n".join(lines + ["| " + " | ".join(str(r.get(c, "")) for c in cols) + " |" for r in rows])


def pipeline_evidence(profile, pipeline_id, update_id):
    """Rows per flow, expectation results and timeline from the pipeline event log (SQL)."""
    src = f"event_log('{pipeline_id}') WHERE origin.update_id = '{update_id}'"
    flows = sql(profile, f"""
        SELECT split(origin.flow_name, '\\\\.')[2] AS flow,
               sum(details:flow_progress.metrics.num_output_rows::bigint) AS rows_written,
               sum(coalesce(details:flow_progress.data_quality.dropped_records::bigint, 0)) AS rows_dropped
        FROM {src} AND event_type = 'flow_progress' AND origin.flow_name IS NOT NULL AND origin.flow_name != 'missingFlowName'
        GROUP BY 1 HAVING rows_written IS NOT NULL ORDER BY 1""")
    quality = sql(profile, f"""
        SELECT flow, x.name AS expectation, sum(x.passed_records) AS passed, sum(x.failed_records) AS failed
        FROM (SELECT split(origin.flow_name, '\\\\.')[2] AS flow,
                     explode(from_json(details:flow_progress.data_quality.expectations,
                       'array<struct<name:string,dataset:string,passed_records:bigint,failed_records:bigint>>')) AS x
              FROM {src} AND event_type = 'flow_progress')
        GROUP BY ALL ORDER BY 1, 2""")
    timeline = sql(profile, f"""
        SELECT timestamp, level, message FROM {src}
        AND event_type IN ('create_update', 'update_progress') ORDER BY timestamp""")
    return "\n".join([
        f"# Lakeflow pipeline update `{update_id}`", "",
        f"Pipeline `{pipeline_id}` (offer-prioritization-medallion). Queried from the pipeline event log with",
        f"`SELECT ... FROM event_log('{pipeline_id}')`.", "",
        "## Rows written per flow (bronze -> silver -> gold)", "", md_table(flows, ["flow", "rows_written", "rows_dropped"]), "",
        "## Data-quality expectations", "",
        "Expectations declared `ON VIOLATION DROP ROW` remove the row; the others keep it and record the violation.", "",
        md_table(quality, ["flow", "expectation", "passed", "failed"]), "",
        "## Update timeline", "", md_table(timeline, ["timestamp", "level", "message"]), ""])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--profile", required=True)
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--out", default="evidence")
    a = ap.parse_args()
    os.makedirs(f"{a.out}/notebooks", exist_ok=True)

    run = cli(a.profile, "jobs", "get-run", a.run_id)
    summary = {"job_id": run["job_id"], "run_id": run["run_id"], "run_page_url": run.get("run_page_url"),
               "state": run["state"], "tasks": []}
    for t in sorted(run["tasks"], key=lambda x: x.get("start_time", 0)):
        entry = {"task_key": t["task_key"], "result_state": t["state"].get("result_state"),
                 "duration_s": round((t.get("end_time", 0) - t.get("start_time", 0)) / 1000)}
        if "notebook_task" in t:
            try:
                entry["notebook_exit"] = json.loads(
                    cli(a.profile, "jobs", "get-run-output", str(t["run_id"])).get("notebook_output", {}).get("result") or "null")
            except json.JSONDecodeError:
                pass
            nb = to_ipynb(notebook_model(cli(a.profile, "jobs", "export-run", str(t["run_id"]), "--views-to-export", "CODE")))
            with open(f"{a.out}/notebooks/{t['task_key']}.ipynb", "w") as f:
                json.dump(nb, f, indent=1)
            with open(f"{a.out}/notebooks/{t['task_key']}.md", "w") as f:
                f.write(to_markdown(t["task_key"], t, nb))
        if "pipeline_task" in t:
            pid = t["pipeline_task"]["pipeline_id"]
            ups = cli(a.profile, "pipelines", "list-updates", pid, "--max-results", "5")
            update = next(u for u in ups.get("updates", []) if u.get("state") == "COMPLETED")
            entry["pipeline_id"], entry["update_id"] = pid, update["update_id"]
            with open(f"{a.out}/pipeline_update.md", "w") as f:
                f.write(pipeline_evidence(a.profile, pid, update["update_id"]))
        summary["tasks"].append(entry)
        print(f"{entry['task_key']:<28} {entry['result_state']:<10} {entry['duration_s']:>5}s")
    with open(f"{a.out}/job_run.json", "w") as f:
        json.dump(summary, f, indent=2, default=str)


if __name__ == "__main__":
    main()
