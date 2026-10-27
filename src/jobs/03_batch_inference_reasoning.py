# Databricks notebook source
# MAGIC %md
# MAGIC # 03 · Batch inference + SHAP explanations + LLM reasoning
# MAGIC
# MAGIC 1. Loads the **`@champion`** model from Unity Catalog and scores all 16 offers for
# MAGIC    every member → `offer_recommendations` (top 5 per member).
# MAGIC 2. For the care-manager work queue (a sample of members), computes per-offer SHAP
# MAGIC    factors and asks a Foundation Model (`databricks-gpt-5-1`, via the Databricks
# MAGIC    serving endpoint) to write a plain-language "why this offer" explanation.
# MAGIC 3. Runs lightweight quality checks on the generated text (guardrails).
# MAGIC 4. Writes `member_offer_recommendations_with_reasoning` (PK member_id, offer_id),
# MAGIC    which is served to the app through Lakebase.

# COMMAND ----------

# MAGIC %pip install lightgbm shap openai mlflow --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %run ../../models/offer_model

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md ## 1 · Score every member against every offer

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md ## 2 · SHAP factors for the care-manager work queue

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md ## 3 · LLM reasoning via the Databricks Foundation Model API

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md ## 4 · Guardrail checks on generated text

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md ## 5 · Serving table consumed by Lakebase + the app

# COMMAND ----------

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

# COMMAND ----------

dbutils.notebook.exit(json.dumps({
    "model_version": str(champion.version), "members_scored": int(len(member_ids)),
    "recommendations": int(len(recs)), "served_rows": int(len(served)), "guardrails": checks,
}))
