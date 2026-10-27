# Databricks notebook source
# MAGIC %md
# MAGIC # 06 · Genie Agent — natural-language access for the business
# MAGIC
# MAGIC Creates (or updates in place) the **"Member Offer Intelligence"** Genie Agent over
# MAGIC the governed gold tables and model outputs, with business instructions, join
# MAGIC specs, example SQL and benchmark questions. Then asks it real business questions
# MAGIC through the Genie Conversation API and prints the generated SQL and answers, so
# MAGIC the run itself is the evidence that the agent works.

# COMMAND ----------

dbutils.widgets.text("catalog", "bhavik_demos")
dbutils.widgets.text("schema", "offer_prioritization")
dbutils.widgets.text("warehouse_id", "c5373a0bb5c8d31b")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
WAREHOUSE_ID = dbutils.widgets.get("warehouse_id")
FQ = f"{CATALOG}.{SCHEMA}"
TITLE = "Member Offer Intelligence"

# COMMAND ----------

import json
import time
import hashlib
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

def gid(label):
    """Stable 32-hex id so re-runs update the same instructions instead of duplicating."""
    return hashlib.md5(label.encode()).hexdigest()

def t(name):
    return f"{FQ}.{name}"

# COMMAND ----------

TEXT_INSTRUCTIONS = [
    "You support a health plan's member engagement and care management teams. All data is synthetic.",
    "Member profile, risk tier, conditions and 12-month utilization: use gold_member_360 (one row per member).",
    "Which offers the model recommends for a member: use offer_recommendations (top 5 per member, rank 1 = best). Join offer_catalog on offer_id for category.",
    "Why an offer was recommended (plain-language reasoning, SHAP factors): use member_offer_recommendations_with_reasoning. It only covers the care-manager work queue sample, not every member.",
    "Claims cost or utilization trends: use gold_claims_monthly and SUM the measures; paid_amount is plan-paid dollars, member_cost is member out-of-pocket.",
    "Outreach performance / response rates by channel: use gold_engagement_performance. Response rate = SUM(responses) / SUM(touches); never average the response_rate column.",
    "Care-manager decisions: use offer_feedback. feedback = 'approved' or 'rejected' for decisions; feedback = 'comment' rows carry free-text notes in feedback_text.",
    "High-risk members means risk_tier = 'High' (risk_score >= 60). Rising risk means risk_tier = 'Medium'.",
    "Round currency to 0 decimals and rates to 1 decimal place as percentages.",
]

EXAMPLES = [
    ("How many members are in each risk tier?",
     f"SELECT risk_tier, COUNT(*) AS members, ROUND(AVG(paid_amount_12m), 0) AS avg_paid_12m FROM {t('gold_member_360')} GROUP BY risk_tier ORDER BY members DESC"),
    ("Which offers are recommended most often as the #1 offer?",
     f"SELECT r.offer_name, c.category, COUNT(*) AS members_rank_1 FROM {t('offer_recommendations')} r JOIN {t('offer_catalog')} c ON r.offer_id = c.offer_id WHERE r.rank = 1 GROUP BY r.offer_name, c.category ORDER BY members_rank_1 DESC"),
    ("What is the outreach response rate by channel?",
     f"SELECT channel, SUM(touches) AS touches, ROUND(100.0 * SUM(responses) / SUM(touches), 1) AS response_rate_pct FROM {t('gold_engagement_performance')} GROUP BY channel ORDER BY response_rate_pct DESC"),
    ("How many high-risk diabetic members have the Diabetes Management Program as their top offer?",
     f"SELECT COUNT(DISTINCT m.member_id) AS members FROM {t('gold_member_360')} m JOIN {t('offer_recommendations')} r ON m.member_id = r.member_id WHERE m.risk_tier = 'High' AND m.has_diabetes = 1 AND r.rank = 1 AND r.offer_id = 'CHRON_001'"),
]

SAMPLE_QUESTIONS = [
    "Which offers are recommended most often as the #1 offer?",
    "What is the outreach response rate by channel?",
    "Show monthly plan-paid claims cost by region for the last 6 months",
    "How many high-risk members have a mental health condition and what is their top recommended offer?",
    "Why was the top offer recommended for member with the highest risk score in the work queue?",
]

BENCHMARKS = [
    ("How many members do we have by plan type?",
     f"SELECT plan_type, COUNT(*) AS members FROM {t('gold_member_360')} GROUP BY plan_type ORDER BY members DESC"),
    ("What was total plan-paid claims cost by claim type?",
     f"SELECT claim_type, ROUND(SUM(paid_amount), 0) AS paid FROM {t('gold_claims_monthly')} GROUP BY claim_type ORDER BY paid DESC"),
    ("What share of members have a cardiovascular condition?",
     f"SELECT ROUND(100.0 * AVG(has_cardiovascular), 1) AS pct FROM {t('gold_member_360')}"),
]

TABLES = {
    "gold_member_360": [
        {"column_name": "member_id", "description": ["Synthetic member identifier (MBR_xxxxxxxx)."]},
        {"column_name": "plan_type", "enable_format_assistance": True, "enable_entity_matching": True},
        {"column_name": "region", "enable_format_assistance": True, "enable_entity_matching": True},
        {"column_name": "risk_tier", "enable_format_assistance": True, "enable_entity_matching": True,
         "synonyms": ["risk level", "risk segment"]},
        {"column_name": "paid_amount_12m", "synonyms": ["cost", "spend", "plan paid"]},
        {"column_name": "er_visits_12m", "synonyms": ["emergency visits", "ER utilization"]},
    ],
    "gold_claims_monthly": [
        {"column_name": "claim_type", "enable_format_assistance": True, "enable_entity_matching": True},
        {"column_name": "diagnosis_category", "enable_format_assistance": True, "enable_entity_matching": True},
        {"column_name": "paid_amount", "synonyms": ["plan paid", "medical cost", "spend"]},
    ],
    "gold_engagement_performance": [
        {"column_name": "channel", "enable_format_assistance": True, "enable_entity_matching": True},
        {"column_name": "engagement_type", "enable_format_assistance": True, "enable_entity_matching": True},
    ],
    "offer_catalog": [
        {"column_name": "offer_name", "enable_format_assistance": True, "enable_entity_matching": True,
         "synonyms": ["program", "offer", "care program"]},
        {"column_name": "category", "enable_format_assistance": True, "enable_entity_matching": True},
        {"column_name": "targeting_rules", "exclude": True},
    ],
    "offer_recommendations": [
        {"column_name": "offer_name", "enable_format_assistance": True, "enable_entity_matching": True},
        {"column_name": "priority_score", "synonyms": ["relevance", "propensity", "score"]},
    ],
    "member_offer_recommendations_with_reasoning": [
        {"column_name": "llm_reasoning", "synonyms": ["why", "explanation", "reason"]},
    ],
    "offer_feedback": [
        {"column_name": "feedback", "enable_format_assistance": True},
    ],
}

JOINS = [
    ("gold_member_360", "offer_recommendations", "member_id", "member_id", "ONE_TO_MANY"),
    ("offer_recommendations", "offer_catalog", "offer_id", "offer_id", "MANY_TO_ONE"),
    ("gold_member_360", "member_offer_recommendations_with_reasoning", "member_id", "member_id", "ONE_TO_MANY"),
    ("offer_recommendations", "offer_feedback", "member_id", "member_id", "ONE_TO_MANY"),
]

space = {
    "version": 2,
    "config": {"sample_questions": sorted(
        [{"id": gid("sq:" + q), "question": [q]} for q in SAMPLE_QUESTIONS], key=lambda x: x["id"])},
    "data_sources": {"tables": sorted(
        [{"identifier": t(n), "column_configs": sorted(c, key=lambda x: x["column_name"])} for n, c in TABLES.items()],
        key=lambda x: x["identifier"])},
    "instructions": {
        "text_instructions": [{"id": gid("ti:main"), "content": [s + "\n" for s in TEXT_INSTRUCTIONS]}],
        "example_question_sqls": sorted(
            [{"id": gid("ex:" + q), "question": [q], "sql": [s]} for q, s in EXAMPLES], key=lambda x: x["id"]),
        "join_specs": sorted([{
            "id": gid(f"join:{l}:{r}"),
            "left": {"identifier": t(l), "alias": l}, "right": {"identifier": t(r), "alias": r},
            "sql": [f"`{l}`.`{lc}` = `{r}`.`{rc}`", f"--rt=FROM_RELATIONSHIP_TYPE_{rel}--"],
        } for l, r, lc, rc, rel in JOINS], key=lambda x: x["id"]),
    },
    "benchmarks": {"questions": sorted([{
        "id": gid("bm:" + q), "question": [q], "answer": [{"format": "SQL", "content": [s]}],
    } for q, s in BENCHMARKS], key=lambda x: x["id"])},
}

DESCRIPTION = ("Ask questions about health-plan members, risk, claims cost, outreach response and the "
               "AI next-best-offer recommendations (with reasoning) and care-manager feedback. Synthetic data.")

existing = w.api_client.do("GET", "/api/2.0/genie/spaces").get("spaces", [])
match = [s for s in existing if s.get("title") == TITLE]
body = {"title": TITLE, "description": DESCRIPTION, "warehouse_id": WAREHOUSE_ID,
        "serialized_space": json.dumps(space)}
if match:
    space_id = match[0]["space_id"]
    w.api_client.do("PATCH", f"/api/2.0/genie/spaces/{space_id}", body=body)
    print(f"Updated Genie Agent '{TITLE}' ({space_id})")
else:
    space_id = w.api_client.do("POST", "/api/2.0/genie/spaces", body=body)["space_id"]
    print(f"Created Genie Agent '{TITLE}' ({space_id})")
host = w.config.host.rstrip("/")
print(f"{host}/genie/rooms/{space_id}")
print(f"{len(TABLES)} tables, {len(EXAMPLES)} example SQLs, {len(JOINS)} join specs, "
      f"{len(SAMPLE_QUESTIONS)} sample questions, {len(BENCHMARKS)} benchmarks")

# COMMAND ----------

# MAGIC %md ## Ask the agent real business questions (Conversation API)

# COMMAND ----------

def ask(question, timeout_s=180):
    base = f"/api/2.0/genie/spaces/{space_id}"
    r = w.api_client.do("POST", f"{base}/start-conversation", body={"content": question})
    cid, mid = r["conversation_id"], r["message_id"]
    deadline = time.time() + timeout_s
    while True:
        msg = w.api_client.do("GET", f"{base}/conversations/{cid}/messages/{mid}")
        if msg.get("status") in ("COMPLETED", "FAILED", "CANCELLED") or time.time() > deadline:
            break
        time.sleep(3)
    out = {"question": question, "status": msg.get("status"), "sql": None, "text": None,
           "columns": [], "rows": []}
    for a in msg.get("attachments", []) or []:
        if a.get("text"):
            out["text"] = a["text"].get("content")
        if a.get("query"):
            out["sql"] = a["query"].get("query")
            res = w.api_client.do(
                "GET", f"{base}/conversations/{cid}/messages/{mid}/attachments/{a['attachment_id']}/query-result")
            stmt = res.get("statement_response", {})
            out["columns"] = [c["name"] for c in stmt.get("manifest", {}).get("schema", {}).get("columns", [])]
            out["rows"] = (stmt.get("result", {}) or {}).get("data_array", [])[:10]
    return out

QUESTIONS = [
    "Which offers are recommended most often as the #1 offer?",
    "What is the outreach response rate by channel?",
    "How many high-risk members have a mental health condition, and which offer is most often ranked #1 for them?",
    "Show total plan-paid claims cost by region",
    "For the member with the highest risk score in the work queue, what are their top offers and why?",
]
transcripts = []
for q in QUESTIONS:
    res = ask(q)
    transcripts.append(res)
    print("=" * 100)
    print(f"Q: {q}\nstatus: {res['status']}")
    if res["sql"]:
        print(f"SQL:\n{res['sql']}")
    if res["columns"]:
        print(" | ".join(res["columns"]))
        for row in res["rows"]:
            print(" | ".join("" if v is None else str(v)[:70] for v in row))
    if res["text"]:
        print(f"Answer: {res['text']}")

# COMMAND ----------

# Strict: an answer counts only if Genie generated SQL that returned rows
ok = sum(1 for r in transcripts if r["status"] == "COMPLETED" and r["sql"] and r["rows"])
clarified = sum(1 for r in transcripts if r["status"] == "COMPLETED" and not r["rows"] and r["text"])
print(f"{ok}/{len(transcripts)} answered with SQL + results; {clarified} asked a clarifying question")
dbutils.notebook.exit(json.dumps({"space_id": space_id, "url": f"{host}/genie/rooms/{space_id}",
                                  "answered": ok, "clarifying": clarified, "asked": len(transcripts)}))
