# Executed notebook: `genie_agent`

- Job run task id: `1091462630722184`
- Result: **SUCCESS**
- Started: 2026-10-27T20:47:33.673000+00:00
- Duration: 100 s

# 06 · Genie Agent — natural-language access for the business

Creates (or updates in place) the **"Member Offer Intelligence"** Genie Agent over
the governed gold tables and model outputs, with business instructions, join
specs, example SQL and benchmark questions. Then asks it real business questions
through the Genie Conversation API and prints the generated SQL and answers, so
the run itself is the evidence that the agent works.

```python
dbutils.widgets.text("catalog", "bhavik_demos")
dbutils.widgets.text("schema", "offer_prioritization")
dbutils.widgets.text("warehouse_id", "c5373a0bb5c8d31b")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
WAREHOUSE_ID = dbutils.widgets.get("warehouse_id")
FQ = f"{CATALOG}.{SCHEMA}"
TITLE = "Member Offer Intelligence"
```

```python
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
```

```python
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
```
Output:
```text
Updated Genie Agent 'Member Offer Intelligence' (01f1d2431e1c1007bbd755dc7cc91fbb)
https://fevm-classic-stable-tp909z.cloud.databricks.com/genie/rooms/01f1d2431e1c1007bbd755dc7cc91fbb
7 tables, 4 example SQLs, 4 join specs, 5 sample questions, 3 benchmarks
```

 ## Ask the agent real business questions (Conversation API)

```python
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
```
Output:
```text
====================================================================================================
Q: Which offers are recommended most often as the #1 offer?
status: COMPLETED
SQL:
SELECT
  `offer_recommendations`.`offer_name`,
  `offer_catalog`.`category`,
  COUNT(*) AS `members_rank_1`
FROM `bhavik_demos`.`offer_prioritization`.`offer_recommendations` AS `offer_recommendations`
JOIN `bhavik_demos`.`offer_prioritization`.`offer_catalog` AS `offer_catalog`
  ON `offer_recommendations`.`offer_id` = `offer_catalog`.`offer_id`
WHERE `offer_recommendations`.`rank` = 1
GROUP BY
  `offer_recommendations`.`offer_name`,
  `offer_catalog`.`category`
ORDER BY `members_rank_1` DESC
offer_name | category | members_rank_1
Heart Health Program | chronic_disease | 1300
Mental Health Support | mental_health | 964
Annual Wellness Visit Reminder | preventive_care | 860
Respiratory Care Program | chronic_disease | 660
HSA/FSA Optimization | cost_savings | 536
Diabetes Management Program | chronic_disease | 458
Generic Drug Switch Program | pharmacy | 108
Cancer Screening Program | preventive_care | 37
Virtual Specialist Consultations | telehealth | 26
In-Network Provider Finder | cost_savings | 14
Answer: Examples include:
- **Heart Health Program** (**chronic_disease**) — **1,300** members had this as their **#1** recommended offer
- **Mental Health Support** (**mental_health**) — **964** members
- **Annual Wellness Visit Reminder** (**preventive_care**) — **860** members
- **Respiratory Care Program** (**chronic_disease**) — **660** members
- **HSA/FSA Optimization** (**cost_savings**) — **536** members

Across all **13** offers shown, chronic disease programs appear especially often near the top, with the highest-count offer being **Heart Health Program**.
====================================================================================================
Q: What is the outreach response rate by channel?
status: COMPLETED
SQL:
SELECT
  `channel`,
  SUM(`touches`) AS `touches`,
  ROUND(try_divide(100.0 * SUM(`responses`),SUM(`touches`)), 1) AS `response_rate_pct`
FROM `bhavik_demos`.`offer_prioritization`.`gold_engagement_performance`
WHERE `channel` IS NOT NULL
GROUP BY `channel`
ORDER BY `response_rate_pct` DESC
channel | touches | response_rate_pct
sms | 7315 | 30.6
mail | 6334 | 30.4
email | 13538 | 30.2
portal | 11031 | 29.9
app | 14133 | 29.8
phone | 10354 | 29.6
Answer: Response rates are very similar across all **6 channels**, ranging from **29.6%** to **30.6%**. Some of the top values are:
- **SMS:** **30.6%** response rate from **7,315** touches
- **Mail:** **30.4%** response rate from **6,334** touches
- **Email:** **30.2%** response rate from **13,538** touches
- **Portal:** **29.9%** response rate from **11,031** touches
- **App:** **29.8%** response rate from **14,133** touches
Overall, no channel stands out strongly; the spread between highest and lowest response rate is just **1.0 percentage point**.
====================================================================================================
Q: How many high-risk members have a mental health condition, and which offer is most often ranked #1 for them?
status: COMPLETED
SQL:
WITH eligible_members AS (
  SELECT DISTINCT `member_id`
  FROM `bhavik_demos`.`offer_prioritization`.`gold_member_360`
  WHERE `risk_tier` = 'High'
    AND `has_mental_health` = 1
),
member_total AS (
  SELECT COUNT(DISTINCT `member_id`) AS `high_risk_mental_health_members`
  FROM eligible_members
),
rank_1_offers AS (
  SELECT
    r.`offer_id`,
    r.`offer_name`,
    COUNT(DISTINCT r.`member_id`) AS `rank_1_members`
  FROM `bhavik_demos`.`offer_prioritization`.`offer_recommendations` r
  INNER JOIN eligible_members e
    ON e.`member_id` = r.`member_id`
  WHERE r.`rank` = 1
    AND r.`offer_id` IS NOT NULL
    AND r.`offer_name` IS NOT NULL
  GROUP BY r.`offer_id`, r.`offer_name`
),
ranked_offers AS (
  SELECT
    `offer_id`,
    `offer_name`,
    `rank_1_members`,
    RANK() OVER (ORDER BY `rank_1_members` DESC, `offer_name` ASC) AS `offer_rank`
  FROM rank_1_offers
)
SELECT
  m.`high_risk_mental_health_members`,
  r.`offer_id`,
  r.`offer_name`,
  r.`rank_1_members`
FROM member_total m
CROSS JOIN ranked_offers r
WHERE r.`offer_rank` = 1
high_risk_mental_health_members | offer_id | offer_name | rank_1_members
37 | MH_001 | Mental Health Support | 25
Answer: There are **37** high-risk members with a mental health condition. The offer most often ranked **#1** for these members is **Mental Health Support** (offer ID **MH_001**), which is ranked #1 for **25** members.
====================================================================================================
Q: Show total plan-paid claims cost by region
status: COMPLETED
SQL:
SELECT `region`, ROUND(SUM(`paid_amount`), 0) AS `total_plan_paid_claims_cost`
FROM `bhavik_demos`.`offer_prioritization`.`gold_claims_monthly`
WHERE `region` IS NOT NULL
GROUP BY `region`
ORDER BY `total_plan_paid_claims_cost` DESC
region | total_plan_paid_claims_cost
Southeast | 7249792.0
Northeast | 6425511.0
Midwest | 6217377.0
West | 6091357.0
Southwest | 5777972.0
Answer: Total plan-paid claims cost is highest in the **Southeast** at **$7,249,792** and lowest in the **Southwest** at **$5,777,972** across the 5 regions shown. Some of the top values are:
- **Southeast**: **$7,249,792**
- **Northeast**: **$6,425,511**
- **Midwest**: **$6,217,377**
- **West**: **$6,091,357**
- **Southwest**: **$5,777,972**
====================================================================================================
Q: For the member with the highest risk score in the work queue, what are their top offers and why?
status: COMPLETED
SQL:
WITH `member_risk` AS (
  SELECT
    `member_id`,
    `risk_score`,
    RANK() OVER (ORDER BY `risk_score` DESC) AS `risk_rank`
  FROM (
    SELECT DISTINCT `member_id`, `risk_score`
    FROM `bhavik_demos`.`offer_prioritization`.`member_offer_recommendations_with_reasoning`
    WHERE `risk_score` IS NOT NULL
  ) t
), `top_members` AS (
  SELECT `member_id`, `risk_score`
  FROM `member_risk`
  WHERE `risk_rank` = 1
)
SELECT
  r.`member_id`,
  tm.`risk_score`,
  r.`rank`,
  r.`offer_name`,
  r.`priority_score`,
  r.`llm_reasoning`,
  r.`shap_factors`
FROM `bhavik_demos`.`offer_prioritization`.`member_offer_recommendations_with_reasoning` r
JOIN `top_members` tm
  ON r.`member_id` = tm.`member_id`
WHERE r.`rank` IS NOT NULL
ORDER BY r.`member_id`, r.`rank` ASC
member_id | risk_score | rank | offer_name | priority_score | llm_reasoning | shap_factors
MBR_6E832AC0 | 72.7599037111 | 1 | Heart Health Program | 98.69 | The Heart Health Program is a strong next step because this member is  | [{"feature": "has_cardiovascular", "value": 1.0, "shap_value": 22.884,
MBR_6E832AC0 | 72.7599037111 | 2 | Cancer Screening Program | 89.12 | The Cancer Screening Program is a good next step because at age 77, st | [{"feature": "risk_score", "value": 72.76, "shap_value": 9.96, "direct
MBR_6E832AC0 | 72.7599037111 | 3 | Annual Wellness Visit Reminder | 71.14 | This member is 77, so staying ahead with preventive care is especially | [{"feature": "chronic_condition_count", "value": 0.0, "shap_value": -3
MBR_6E832AC0 | 72.7599037111 | 4 | Care Navigator Assignment | 70.0 | The Care Navigator Assignment program can help this member better coor | [{"feature": "unique_diagnosis_count", "value": 5.0, "shap_value": 9.3
MBR_6E832AC0 | 72.7599037111 | 5 | HSA/FSA Optimization | 66.77 | The HSA/FSA Optimization program can help this member make the most of | [{"feature": "remaining_deductible_pct", "value": 0.5, "shap_value": -
Answer: The highest-risk member in the work queue is **MBR_6E832AC0** with a risk score of **72.8**. Their top offers are:
- **1. Heart Health Program** — recommended because the member has a heart-related condition, is **77** years old, and their higher risk level suggests added monitoring and care coordination could help prevent complications.
- **2. Cancer Screening Program** — recommended due to the member’s **age (77)** and higher risk level, with support to stay current on appropriate screenings and follow-up.
- **3. Annual Wellness Visit Reminder** — recommended because preventive care is important at **77**, especially to review heart health, medications, and early signs of change.
- **4. Care Navigator Assignment** — recommended because the member has several diagnoses and more complex care needs, so help coordinating appointments, tests, and follow-up may be valuable.
- **5. HSA/FSA Optimization** — recommended to help manage ongoing healthcare expenses and make better use of available pre-tax health spending funds.
Across these 5 offers, the main themes are the member’s **heart-related condition**, **older age**, **high risk**, and need for stronger care coordination and preventive support.
```

```python
# Strict: an answer counts only if Genie generated SQL that returned rows
ok = sum(1 for r in transcripts if r["status"] == "COMPLETED" and r["sql"] and r["rows"])
clarified = sum(1 for r in transcripts if r["status"] == "COMPLETED" and not r["rows"] and r["text"])
print(f"{ok}/{len(transcripts)} answered with SQL + results; {clarified} asked a clarifying question")
dbutils.notebook.exit(json.dumps({"space_id": space_id, "url": f"{host}/genie/rooms/{space_id}",
                                  "answered": ok, "clarifying": clarified, "asked": len(transcripts)}))
```
Output:
```text
{"space_id": "01f1d2431e1c1007bbd755dc7cc91fbb", "url": "https://fevm-classic-stable-tp909z.cloud.databricks.com/genie/rooms/01f1d2431e1c1007bbd755dc7cc91fbb", "answered": 5, "clarifying": 0, "asked": 5}
```
