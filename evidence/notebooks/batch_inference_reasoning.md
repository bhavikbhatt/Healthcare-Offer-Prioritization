# Executed notebook: `batch_inference_reasoning`

- Job run task id: `414084989226100`
- Result: **SUCCESS**
- Started: 2026-10-27T20:23:26.575000+00:00
- Duration: 205 s

# 03 · Batch inference + SHAP explanations + LLM reasoning

1. Loads the **`@champion`** model from Unity Catalog and scores all 16 offers for
   every member → `offer_recommendations` (top 5 per member).
2. For the care-manager work queue (a sample of members), computes per-offer SHAP
   factors and asks a Foundation Model (`databricks-gpt-5-1`, via the Databricks
   serving endpoint) to write a plain-language "why this offer" explanation.
3. Runs lightweight quality checks on the generated text (guardrails).
4. Writes `member_offer_recommendations_with_reasoning` (PK member_id, offer_id),
   which is served to the app through Lakebase.

```python
%pip install lightgbm shap openai mlflow --quiet
```
Output:
```text
Note: you may need to restart the kernel using %restart_python or dbutils.library.restartPython() to use updated packages.
```

```python
dbutils.library.restartPython()
```

```python
dbutils.widgets.text("catalog", "bhavik_demos")
dbutils.widgets.text("schema", "offer_prioritization")
dbutils.widgets.text("llm_sample", "100")
dbutils.widgets.text("llm_endpoint", "databricks-gpt-5-1")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FQ = f"{CATALOG}.{SCHEMA}"
UC_MODEL = f"{FQ}.healthcare_offer_prioritizer"
LLM_SAMPLE = int(dbutils.widgets.get("llm_sample"))
LLM_ENDPOINT = dbutils.widgets.get("llm_endpoint")
```

```python
%run ../../models/offer_model
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python
import json
import time
import numpy as np
import pandas as pd
from datetime import datetime, timezone
import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_registry_uri("databricks-uc")
champion = MlflowClient().get_model_version_by_alias(UC_MODEL, "champion")
pipeline = mlflow.sklearn.load_model(f"models:/{UC_MODEL}@champion")
meta = mlflow.artifacts.load_dict(f"runs:/{champion.run_id}/model_metadata.json")
offer_ids, feature_names = meta["offer_ids"], meta["feature_names"]
print(f"Loaded {UC_MODEL} v{champion.version} @champion  ({len(feature_names)} features, {len(offer_ids)} offers)")

features = spark.table(f"{FQ}.member_features").toPandas()
batch_id = features["batch_id"].iloc[0]
X = features[feature_names]
member_ids = features["member_id"].values
catalog = OfferCatalog()
offer_name = {o["offer_id"]: o["name"] for o in catalog.offers}
offer_category = {o["offer_id"]: o["category"] for o in catalog.offers}
```
Output:
```text
Loaded bhavik_demos.offer_prioritization.healthcare_offer_prioritizer v3 @champion  (96 features, 16 offers)
```

 ## 1 · Score every member against every offer

```python
t0 = time.time()
scores = np.clip(pipeline.predict(X), 0, 100)
print(f"Scored {scores.shape[0]:,} members x {scores.shape[1]} offers in {time.time()-t0:.1f}s")

TOP_N, MIN_SCORE = 5, 25.0
rows = []
for i, m in enumerate(member_ids):
    order = np.argsort(-scores[i])
    rank = 0
    for j in order:
        if scores[i, j] < MIN_SCORE or rank == TOP_N:
            break
        rank += 1
        rows.append({"member_id": m, "rank": rank, "offer_id": offer_ids[j],
                     "offer_name": offer_name[offer_ids[j]], "category": offer_category[offer_ids[j]],
                     "priority_score": round(float(scores[i, j]), 2)})
recs = pd.DataFrame(rows)
recs["model_version"] = str(champion.version)
recs["batch_id"] = batch_id
recs["scored_at"] = datetime.now(timezone.utc).isoformat()

spark.createDataFrame(recs).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{FQ}.offer_recommendations")
print(f"wrote {FQ}.offer_recommendations: {len(recs):,} rows, {recs.member_id.nunique():,} members")
print("\nTop-ranked offer distribution (rank 1):")
print(recs[recs["rank"] == 1]["offer_name"].value_counts().to_string())
```
Output:
```text
Scored 4,980 members x 16 offers in 0.7s
wrote bhavik_demos.offer_prioritization.offer_recommendations: 24,900 rows, 4,980 members

Top-ranked offer distribution (rank 1):
offer_name
Heart Health Program                1300
Mental Health Support                964
Annual Wellness Visit Reminder       860
Respiratory Care Program             660
HSA/FSA Optimization                 536
Diabetes Management Program          458
Generic Drug Switch Program          108
Cancer Screening Program              37
Virtual Specialist Consultations      26
In-Network Provider Finder            14
Virtual Primary Care                   8
Care Navigator Assignment              8
Mail-Order Pharmacy                    1
```

 ## 2 · SHAP factors for the care-manager work queue

```python
import shap

sample_ids = pd.Series(member_ids).sample(LLM_SAMPLE, random_state=7).tolist()
sample_recs = recs[recs["member_id"].isin(sample_ids)].copy()
idx_of = {m: i for i, m in enumerate(member_ids)}
X_scaled = pipeline.named_steps["scaler"].transform(X)
sample_rows = np.array([idx_of[m] for m in sample_ids])

t0 = time.time()
estimators = pipeline.named_steps["model"].estimators_
shap_by_offer = {oid: shap.TreeExplainer(est).shap_values(X_scaled[sample_rows])
                 for oid, est in zip(offer_ids, estimators)}
print(f"SHAP for {len(sample_ids)} members x {len(offer_ids)} offers in {time.time()-t0:.1f}s")

pos_in_sample = {m: k for k, m in enumerate(sample_ids)}
def top_factors(member_id, offer_id, k=5):
    sv = shap_by_offer[offer_id][pos_in_sample[member_id]]
    top = np.argsort(-np.abs(sv))[:k]
    raw = X.iloc[idx_of[member_id]]
    return [{"feature": feature_names[t], "value": round(float(raw.iloc[t]), 2),
             "shap_value": round(float(sv[t]), 3),
             "direction": "increases" if sv[t] > 0 else "decreases"} for t in top]

sample_recs["shap_factors"] = [json.dumps(top_factors(r.member_id, r.offer_id))
                               for r in sample_recs.itertuples()]
ex = sample_recs.iloc[0]
print(f"\nExample — {ex.member_id} / {ex.offer_name} (score {ex.priority_score}):")
for f in json.loads(ex.shap_factors):
    print(f"  {f['feature']:<32} value={f['value']:<10} shap={f['shap_value']:+.3f} ({f['direction']})")
```
Output:
```text
SHAP for 100 members x 16 offers in 4.2s

Example — MBR_AC0F1A16 / Mental Health Support (score 68.16):
  has_mental_health                value=0.0        shap=-8.354 (decreases)
  age                              value=20.0       shap=+4.169 (increases)
  engagement_trend                 value=1.6        shap=+0.766 (increases)
  age_group_encoded                value=0.0        shap=+0.618 (increases)
  has_gastrointestinal             value=1.0        shap=-0.433 (decreases)
```

 ## 3 · LLM reasoning via the Databricks Foundation Model API

```python
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
host = "https://" + spark.conf.get("spark.databricks.workspaceUrl").replace("https://", "")
llm = OpenAI(api_key=ctx.apiToken().get(), base_url=f"{host}/serving-endpoints")

PROFILE_COLS = ["age", "risk_score", "chronic_condition_count", "total_claims_count", "total_engagements",
                "has_diabetes", "has_cardiovascular", "has_respiratory", "has_mental_health",
                "is_complex_patient", "is_senior", "tenure_months", "high_risk_flag",
                "avg_utilization_rate", "avg_response_rate"]
prof = features.set_index("member_id")

def profile(m):
    p = prof.loc[m]
    return {c: (float(p[c]) if c in p.index and pd.notna(p[c]) else 0.0) for c in PROFILE_COLS}

SYSTEM = ("You are a health plan care-management assistant. You write short, warm, specific "
          "explanations a care manager can read to a member. Never mention scores, models, "
          "SHAP, features or probabilities. Never give a diagnosis or medical advice.")

def reason(member_id, offer_id, factors):
    p = profile(member_id)
    ftxt = "\n".join(f"- {f['feature'].replace('_', ' ')} = {f['value']} ({f['direction']} relevance)"
                     for f in factors)
    prompt = (f"In 3-4 sentences, explain why the '{offer_name[offer_id]}' program is a good next step "
              f"for this member. No bullet points.\n\nMEMBER: age {p['age']:.0f}, risk score {p['risk_score']:.0f}/100, "
              f"{p['chronic_condition_count']:.0f} chronic conditions, diabetes={bool(p['has_diabetes'])}, "
              f"cardiovascular={bool(p['has_cardiovascular'])}, respiratory={bool(p['has_respiratory'])}, "
              f"mental health={bool(p['has_mental_health'])}, complex care={bool(p['is_complex_patient'])}, "
              f"tenure {p['tenure_months']:.0f} months.\nWHAT DROVE THE RECOMMENDATION:\n{ftxt}")
    for attempt in range(3):
        try:
            r = llm.chat.completions.create(
                model=LLM_ENDPOINT, max_tokens=220, temperature=0.4,
                messages=[{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}])
            return r.choices[0].message.content.strip(), "llm"
        except Exception as e:
            err = str(e)[:200]
            time.sleep(2 ** attempt)
    return (f"The {offer_name[offer_id]} aligns with this member's recent care and engagement history.",
            f"fallback: {err}")

t0 = time.time()
with ThreadPoolExecutor(max_workers=8) as pool:
    out = list(pool.map(lambda r: reason(r.member_id, r.offer_id, json.loads(r.shap_factors)),
                        sample_recs.itertuples()))
sample_recs["llm_reasoning"] = [o[0] for o in out]
sample_recs["reasoning_source"] = [o[1] if o[1] == "llm" else "fallback" for o in out]
print(f"Generated {len(out)} explanations with {LLM_ENDPOINT} in {time.time()-t0:.1f}s")
```
Output:
```text
Generated 500 explanations with databricks-gpt-5-1 in 128.0s
```

 ## 4 · Guardrail checks on generated text

```python
import re
txt = sample_recs["llm_reasoning"]
checks = {
    "llm_generated_pct": float((sample_recs["reasoning_source"] == "llm").mean()),
    "mentions_score_or_model_pct": float(txt.str.contains(r"\bscore\b|\bmodel\b|\bSHAP\b|probabilit", case=False, regex=True).mean()),
    "names_the_program_pct": float(np.mean([offer_name[o].split()[0].lower() in t.lower()
                                            for o, t in zip(sample_recs.offer_id, txt)])),
    "median_words": float(txt.str.split().str.len().median()),
    "over_120_words_pct": float((txt.str.split().str.len() > 120).mean()),
}
print(json.dumps(checks, indent=2))

for r in sample_recs.sort_values(["member_id", "rank"]).head(3).itertuples():
    print(f"\n[{r.member_id}] #{r.rank} {r.offer_name} ({r.priority_score})\n  {r.llm_reasoning}")
```
Output:
```text
{
  "llm_generated_pct": 1.0,
  "mentions_score_or_model_pct": 0.0,
  "names_the_program_pct": 0.938,
  "median_words": 89.0,
  "over_120_words_pct": 0.0
}

[MBR_014716CC] #1 HSA/FSA Optimization (80.24)
  This member is still paying a large share of costs toward their deductible, so learning how to better use an HSA or FSA can help them keep more money in their pocket. Their medical spending so far shows they do use care, but not constantly, which is a good time to plan ahead for expected and unexpected expenses. The program can walk them through which expenses are eligible, how much to set aside, and how to time contributions so they get the most tax benefit. This can make future doctor visits, prescriptions, or procedures more affordable and predictable.

[MBR_014716CC] #2 Annual Wellness Visit Reminder (70.28)
  An Annual Wellness Visit Reminder is a good next step because this member hasn’t used their health benefits or seen a provider in quite some time. Even though they’re young and don’t have ongoing health conditions, a yearly check-in can help catch small issues early and keep them on track with preventive care. This visit also gives them a chance to ask questions and update their health history, which supports better care in the future.

[MBR_014716CC] #3 Mental Health Support (68.9)
  The Mental Health Support program can help this member build skills to manage everyday stress, life changes, or worries before they become harder to handle. At age 35, many people are juggling work, family, and financial responsibilities, and having extra emotional support can make a real difference. The program offers a safe place to talk through concerns and learn practical coping tools, even if the member has never had mental health treatment before. It can also help them feel more connected and supported, especially since they haven’t engaged with their health plan recently.
```

 ## 5 · Serving table consumed by Lakebase + the app

```python
served = sample_recs.merge(
    features[["member_id"] + [c for c in PROFILE_COLS if c in features.columns]]
    .rename(columns={"total_claims_count": "total_claims"}),
    on="member_id", how="left")
for c in PROFILE_COLS:
    c = "total_claims" if c == "total_claims_count" else c
    if c not in served.columns:
        served[c] = 0
for c in ["has_diabetes", "has_cardiovascular", "has_respiratory", "has_mental_health", "is_complex_patient"]:
    served[c] = served[c].fillna(0).astype(bool)
for c in ["age", "chronic_condition_count", "total_claims", "total_engagements", "tenure_months",
          "is_senior", "high_risk_flag"]:
    served[c] = served[c].fillna(0).astype(int)
served["generated_at"] = datetime.now(timezone.utc).isoformat()

COLS = ["member_id", "offer_id", "offer_name", "rank", "priority_score", "llm_reasoning", "shap_factors",
        "age", "risk_score", "chronic_condition_count", "total_claims", "total_engagements",
        "has_diabetes", "has_cardiovascular", "has_respiratory", "has_mental_health", "is_complex_patient",
        "tenure_months", "is_senior", "high_risk_flag", "avg_utilization_rate", "avg_response_rate",
        "reasoning_source", "generated_at", "model_version", "batch_id"]
TABLE = f"{FQ}.member_offer_recommendations_with_reasoning"
spark.createDataFrame(served[COLS].sort_values(["member_id", "rank"])) \
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TABLE)
spark.sql(f"ALTER TABLE {TABLE} ALTER COLUMN member_id SET NOT NULL")
spark.sql(f"ALTER TABLE {TABLE} ALTER COLUMN offer_id SET NOT NULL")
spark.sql(f"ALTER TABLE {TABLE} ADD CONSTRAINT pk_member_offer PRIMARY KEY (member_id, offer_id)")
print(f"wrote {TABLE}: {len(served):,} rows / {served.member_id.nunique()} members")
display(spark.sql(f"SELECT member_id, rank, offer_name, priority_score, left(llm_reasoning, 120) AS reasoning "
                  f"FROM {TABLE} ORDER BY member_id, rank LIMIT 10"))
```
Output:
```text
wrote bhavik_demos.offer_prioritization.member_offer_recommendations_with_reasoning: 500 rows / 100 members

member_id | rank | offer_name | priority_score | reasoning
--- | --- | --- | --- | ---
MBR_014716CC | 1 | HSA/FSA Optimization | 80.24 | This member is still paying a large share of costs toward their deductible, so learning how to better use an HSA or FSA 
MBR_014716CC | 2 | Annual Wellness Visit Reminder | 70.28 | An Annual Wellness Visit Reminder is a good next step because this member hasn’t used their health benefits or seen a pr
MBR_014716CC | 3 | Mental Health Support | 68.9 | The Mental Health Support program can help this member build skills to manage everyday stress, life changes, or worries 
MBR_014716CC | 4 | Stress Management Workshop | 61.61 | The Stress Management Workshop is a good next step because it focuses on building everyday skills to handle pressure bef
MBR_014716CC | 5 | Respiratory Care Program | 56.87 | The Respiratory Care Program is a good next step because it offers education and support to help the member stay ahead o
MBR_01C27037 | 1 | Annual Wellness Visit Reminder | 69.95 | This member is at an age where regular preventive checkups can help catch potential health issues early, even if they’re
MBR_01C27037 | 2 | Mental Health Support | 67.28 | This program is a good next step because it can help the member build emotional resilience and coping skills before stre
MBR_01C27037 | 3 | HSA/FSA Optimization | 66.68 | The HSA/FSA Optimization program is a good fit because this member is already highly engaged, clicking on several offers
MBR_01C27037 | 4 | Stress Management Workshop | 64.68 | The Stress Management Workshop is a good next step because this member has already been actively engaging with their hea
MBR_01C27037 | 5 | Virtual Primary Care | 63.03 | Virtual Primary Care is a good fit because this member is already comfortable using telehealth and engaging with the app
```

```python
dbutils.notebook.exit(json.dumps({
    "model_version": str(champion.version), "members_scored": int(len(member_ids)),
    "recommendations": int(len(recs)), "served_rows": int(len(served)), "guardrails": checks,
}))
```
Output:
```text
{"model_version": "3", "members_scored": 4980, "recommendations": 24900, "served_rows": 500, "guardrails": {"llm_generated_pct": 1.0, "mentions_score_or_model_pct": 0.0, "names_the_program_pct": 0.938, "median_words": 89.0, "over_120_words_pct": 0.0}}
```
