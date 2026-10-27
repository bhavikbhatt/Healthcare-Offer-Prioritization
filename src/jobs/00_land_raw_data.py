# Databricks notebook source
# MAGIC %md
# MAGIC # 00 · Land raw source extracts
# MAGIC
# MAGIC Simulates the health plan's nightly source-system extracts (eligibility, claims,
# MAGIC benefits accumulators, CRM engagement) by generating **synthetic** data and landing
# MAGIC it as JSON files in the Unity Catalog Volume `raw_landing`. Each run is one extract
# MAGIC batch (`batch_id`). The Lakeflow pipeline picks the files up incrementally with
# MAGIC Auto Loader — nothing downstream reads these pandas frames directly.
# MAGIC
# MAGIC No real member data is used anywhere in this project.

# COMMAND ----------

dbutils.widgets.text("catalog", "bhavik_demos")
dbutils.widgets.text("schema", "offer_prioritization")
dbutils.widgets.text("n_members", "5000")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
N_MEMBERS = int(dbutils.widgets.get("n_members"))
LANDING = f"/Volumes/{CATALOG}/{SCHEMA}/raw_landing"

# COMMAND ----------

# MAGIC %run ../../data/generate_synthetic_data

# COMMAND ----------

import os
import json
from datetime import datetime, timezone

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.raw_landing "
          "COMMENT 'Raw synthetic source extracts landed as JSON for Lakeflow Auto Loader ingestion'")

batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
members_df, claims_df, benefits_df, engagement_df = generate_all_data(n_members=N_MEMBERS, seed=42)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Realistic source defects
# MAGIC Real eligibility / claims / CRM extracts are never clean. A small, documented share
# MAGIC of records (~0.5%) is deliberately corrupted so the silver-layer data-quality
# MAGIC expectations in the Lakeflow pipeline have real violations to drop or flag.

# COMMAND ----------

import numpy as np

rng = np.random.default_rng(7)

def corrupt(pdf, frac, col, value):
    """Overwrite `col` on a random `frac` of rows; `value` may be a function of the old values."""
    idx = pdf.sample(frac=frac, random_state=int(rng.integers(1_000_000))).index
    pdf[col] = pdf[col].astype(object) if value is None else pdf[col]
    pdf.loc[idx, col] = value(pdf.loc[idx]) if callable(value) else value
    return len(idx)

defects = {
    "members.age_out_of_range (drop)":         corrupt(members_df, 0.004, "age", 150),
    "members.unknown_plan_type (warn)":         corrupt(members_df, 0.002, "plan_type", "UNKNOWN"),
    "claims.negative_amount (drop)":            corrupt(claims_df, 0.004, "claim_amount", lambda d: -d["claim_amount"]),
    "claims.missing_member_id (drop)":          corrupt(claims_df, 0.002, "member_id", None),
    "claims.paid_above_billed (warn)":          corrupt(claims_df, 0.003, "paid_amount", lambda d: d["claim_amount"] * 1.4),
    "benefits.utilization_over_100pct (drop)":  corrupt(benefits_df, 0.002, "utilization_rate", 1.5),
    "engagement.unknown_channel (drop)":        corrupt(engagement_df, 0.003, "channel", "fax"),
}
for k, v in defects.items():
    print(f"injected {v:>4} defective rows: {k}")

# COMMAND ----------

landed = {}
for source, pdf in [
    ("members", members_df),
    ("claims", claims_df),
    ("benefits_utilization", benefits_df),
    ("engagement_history", engagement_df),
]:
    out_dir = f"{LANDING}/{source}"
    os.makedirs(out_dir, exist_ok=True)
    path = f"{out_dir}/batch_{batch_id}.json"
    pdf = pdf.drop(columns=["created_at"], errors="ignore").assign(batch_id=batch_id)
    pdf.to_json(path, orient="records", lines=True, date_format="iso")
    landed[source] = {"rows": len(pdf), "bytes": os.path.getsize(path), "path": path}
    print(f"{source:<22} {len(pdf):>8,} rows  {os.path.getsize(path)/1e6:6.2f} MB  -> {path}")

# COMMAND ----------

# Peek at the raw landed file exactly as the pipeline will see it
with open(landed["claims"]["path"]) as f:
    for _ in range(3):
        print(f.readline().strip())

# COMMAND ----------

dbutils.jobs.taskValues.set("batch_id", batch_id)
dbutils.notebook.exit(json.dumps({"batch_id": batch_id, "landed": landed, "injected_defects": defects}))
