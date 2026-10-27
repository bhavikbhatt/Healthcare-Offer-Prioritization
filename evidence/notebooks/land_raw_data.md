# Executed notebook: `land_raw_data`

- Job run task id: `1004433180209717`
- Result: **SUCCESS**
- Started: 2026-10-27T20:18:58.039000+00:00
- Duration: 37 s

# 00 · Land raw source extracts

Simulates the health plan's nightly source-system extracts (eligibility, claims,
benefits accumulators, CRM engagement) by generating **synthetic** data and landing
it as JSON files in the Unity Catalog Volume `raw_landing`. Each run is one extract
batch (`batch_id`). The Lakeflow pipeline picks the files up incrementally with
Auto Loader — nothing downstream reads these pandas frames directly.

No real member data is used anywhere in this project.

```python
dbutils.widgets.text("catalog", "bhavik_demos")
dbutils.widgets.text("schema", "offer_prioritization")
dbutils.widgets.text("n_members", "5000")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
N_MEMBERS = int(dbutils.widgets.get("n_members"))
LANDING = f"/Volumes/{CATALOG}/{SCHEMA}/raw_landing"
```

```python
%run ../../data/generate_synthetic_data
```

```python

```

```python

```

```python

```

```python
import os
import json
from datetime import datetime, timezone

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA}.raw_landing "
          "COMMENT 'Raw synthetic source extracts landed as JSON for Lakeflow Auto Loader ingestion'")

batch_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
members_df, claims_df, benefits_df, engagement_df = generate_all_data(n_members=N_MEMBERS, seed=42)
```
Output:
```text
🏥 Generating Healthcare Offer Prioritization Demo Data
============================================================

👥 Generating 5,000 members...
   ✓ Created 5,000 member records

📋 Generating claims (last 730 days)...
   ✓ Created 21,350 claim records

💊 Generating benefits utilization...
   ✓ Created 35,000 benefit records

📱 Generating engagement history (last 365 days)...
   ✓ Created 63,175 engagement records

============================================================
✅ Data generation complete!
```

### Realistic source defects
Real eligibility / claims / CRM extracts are never clean. A small, documented share
of records (~0.5%) is deliberately corrupted so the silver-layer data-quality
expectations in the Lakeflow pipeline have real violations to drop or flag.

```python
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
```
Output:
```text
injected   20 defective rows: members.age_out_of_range (drop)
injected   10 defective rows: members.unknown_plan_type (warn)
injected   85 defective rows: claims.negative_amount (drop)
injected   43 defective rows: claims.missing_member_id (drop)
injected   64 defective rows: claims.paid_above_billed (warn)
injected   70 defective rows: benefits.utilization_over_100pct (drop)
injected  190 defective rows: engagement.unknown_channel (drop)
```

```python
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
```
Output:
```text
members                   5,000 rows    1.40 MB  -> /Volumes/bhavik_demos/offer_prioritization/raw_landing/members/batch_20261027T201919Z.json
claims                   21,350 rows    8.06 MB  -> /Volumes/bhavik_demos/offer_prioritization/raw_landing/claims/batch_20261027T201919Z.json
benefits_utilization     35,000 rows    8.98 MB  -> /Volumes/bhavik_demos/offer_prioritization/raw_landing/benefits_utilization/batch_20261027T201919Z.json
engagement_history       63,175 rows   16.31 MB  -> /Volumes/bhavik_demos/offer_prioritization/raw_landing/engagement_history/batch_20261027T201919Z.json
```

```python
# Peek at the raw landed file exactly as the pipeline will see it
with open(landed["claims"]["path"]) as f:
    for _ in range(3):
        print(f.readline().strip())
```
Output:
```text
{"claim_id":"CLM_B75EA60F-B8A","member_id":"MBR_F15087A6","claim_date":"2024-12-13T20:19:19.382","claim_type":"primary_care","diagnosis_category":"general_wellness","claim_amount":201.37,"allowed_amount":169.8398034109,"paid_amount":281.918,"member_responsibility":56.0288279787,"provider_type":"primary_care","in_network":false,"status":"paid","batch_id":"20261027T201919Z"}
{"claim_id":"CLM_CA2975EB-F0F","member_id":"MBR_F15087A6","claim_date":"2026-06-03T20:19:19.382","claim_type":"outpatient","diagnosis_category":"general_wellness","claim_amount":571.95,"allowed_amount":502.8866363568,"paid_amount":484.3027568388,"member_responsibility":121.4723066593,"provider_type":"ambulatory","in_network":true,"status":"pending","batch_id":"20261027T201919Z"}
{"claim_id":"CLM_31BBE5C1-87B","member_id":"MBR_F15087A6","claim_date":"2025-08-07T20:19:19.382","claim_type":"lab","diagnosis_category":"diabetes","claim_amount":548.78,"allowed_amount":387.3257819243,"paid_amount":288.430712469,"member_responsibility":145.3475629688,"provider_type":"laboratory","in_network":true,"status":"paid","batch_id":"20261027T201919Z"}
```

```python
dbutils.jobs.taskValues.set("batch_id", batch_id)
dbutils.notebook.exit(json.dumps({"batch_id": batch_id, "landed": landed, "injected_defects": defects}))
```
Output:
```text
{"batch_id": "20261027T201919Z", "landed": {"members": {"rows": 5000, "bytes": 1397304, "path": "/Volumes/bhavik_demos/offer_prioritization/raw_landing/members/batch_20261027T201919Z.json"}, "claims": {"rows": 21350, "bytes": 8058557, "path": "/Volumes/bhavik_demos/offer_prioritization/raw_landing/claims/batch_20261027T201919Z.json"}, "benefits_utilization": {"rows": 35000, "bytes": 8975284, "path": "/Volumes/bhavik_demos/offer_prioritization/raw_landing/benefits_utilization/batch_20261027T201919Z.json"}, "engagement_history": {"rows": 63175, "bytes": 16314799, "path": "/Volumes/bhavik_demos/offer_prioritization/raw_landing/engagement_history/batch_20261027T201919Z.json"}}, "injected_defects": {"members.age_out_of_range (drop)": 20, "members.unknown_plan_type (warn)": 10, "claims.negative_amount (drop)": 85, "claims.missing_member_id (drop)": 43, "claims.paid_above_billed (warn)": 64, "benefits.utilization_over_100pct (drop)": 70, "engagement.unknown_channel (drop)": 190}}
```
