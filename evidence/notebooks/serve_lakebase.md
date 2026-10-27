# Executed notebook: `serve_lakebase`

- Job run task id: `195265993185385`
- Result: **SUCCESS**
- Started: 2026-10-27T20:26:52.362000+00:00
- Duration: 29 s

# 04 · Operational serving in Lakebase

Publishes the governed Delta table `member_offer_recommendations_with_reasoning`
into the Lakebase (managed Postgres) table `offer_prioritization.lakebase_offers`
that the **Offer Management Console** app reads, then measures member-lookup
latency the way the app issues it (indexed point lookup by `member_id`).

Credentials are short-lived OAuth tokens minted for this job's identity —
no passwords or secrets are stored anywhere.

```python
%pip install "psycopg[binary]" "databricks-sdk>=0.81.0" --quiet
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
dbutils.widgets.text("lakebase_endpoint", "projects/offer-prioritization/branches/production/endpoints/primary")
dbutils.widgets.text("lakebase_host", "ep-sweet-shadow-d2du2zdd.database.us-east-1.cloud.databricks.com")
dbutils.widgets.text("app_sp_client_id", "f9bec3ef-421b-4a07-bf08-46c6df20c976")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
SOURCE_TABLE = f"{CATALOG}.{SCHEMA}.member_offer_recommendations_with_reasoning"
PG_HOST = dbutils.widgets.get("lakebase_host")
ENDPOINT = dbutils.widgets.get("lakebase_endpoint")
APP_SP = dbutils.widgets.get("app_sp_client_id")
PG_SCHEMA, PG_TABLE = "offer_prioritization", "lakebase_offers"
```

```python
import json
import time
import numpy as np
import pandas as pd
import psycopg
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
token = w.postgres.generate_database_credential(endpoint=ENDPOINT).token
pg_user = spark.sql("SELECT current_user()").first()[0]
conn = psycopg.connect(host=PG_HOST, port=5432, dbname="databricks_postgres",
                       user=pg_user, password=token, sslmode="require", autocommit=True)
cur = conn.cursor()
cur.execute("SELECT version()")
print(f"Connected to Lakebase {ENDPOINT} as {pg_user}\n  {cur.fetchone()[0][:80]}")
```
Output:
```text
Connected to Lakebase projects/offer-prioritization/branches/production/endpoints/primary as bhavik.bhatt@databricks.com
  PostgreSQL 17.11 (11592b3) on x86_64-pc-linux-gnu, compiled by gcc (Ubuntu 13.3.
```

```python
COLS = [
    ("member_id", "text"), ("offer_id", "text"), ("offer_name", "text"), ("rank", "integer"),
    ("priority_score", "double precision"), ("llm_reasoning", "text"), ("shap_factors", "text"),
    ("age", "integer"), ("risk_score", "double precision"), ("chronic_condition_count", "integer"),
    ("total_claims", "integer"), ("total_engagements", "integer"), ("has_diabetes", "boolean"),
    ("has_cardiovascular", "boolean"), ("has_respiratory", "boolean"), ("has_mental_health", "boolean"),
    ("is_complex_patient", "boolean"), ("tenure_months", "integer"), ("is_senior", "integer"),
    ("high_risk_flag", "integer"), ("avg_utilization_rate", "double precision"),
    ("avg_response_rate", "double precision"), ("generated_at", "text"), ("model_version", "text"),
]
names = [c for c, _ in COLS]
pdf = spark.table(SOURCE_TABLE).select(*names).toPandas()
print(f"Read {len(pdf):,} rows / {pdf.member_id.nunique()} members from {SOURCE_TABLE}")

# to_dict gives native Python types (psycopg cannot adapt numpy scalars)
rows = [tuple(None if (isinstance(v, float) and np.isnan(v)) else v for v in rec.values())
        for rec in pdf.to_dict("records")]

# Atomic swap: load a staging table, then rename inside one transaction so the
# app never sees an empty or half-loaded table.
conn.autocommit = False
cur.execute(f"CREATE SCHEMA IF NOT EXISTS {PG_SCHEMA}")
cur.execute(f"DROP TABLE IF EXISTS {PG_SCHEMA}.{PG_TABLE}_staging")
cur.execute(f"CREATE TABLE {PG_SCHEMA}.{PG_TABLE}_staging ("
            + ", ".join(f'"{c}" {t}' for c, t in COLS)
            + ", PRIMARY KEY (member_id, offer_id))")
with cur.copy(f"COPY {PG_SCHEMA}.{PG_TABLE}_staging ({', '.join(names)}) FROM STDIN") as copy:
    for r in rows:
        copy.write_row(r)
cur.execute(f"DROP TABLE IF EXISTS {PG_SCHEMA}.{PG_TABLE}")
cur.execute(f"ALTER TABLE {PG_SCHEMA}.{PG_TABLE}_staging RENAME TO {PG_TABLE}")
cur.execute(f"CREATE INDEX IF NOT EXISTS idx_{PG_TABLE}_member ON {PG_SCHEMA}.{PG_TABLE} (member_id)")
cur.execute(f'GRANT USAGE ON SCHEMA {PG_SCHEMA} TO "{APP_SP}"')
cur.execute(f'GRANT SELECT ON {PG_SCHEMA}.{PG_TABLE} TO "{APP_SP}"')
conn.commit()
conn.autocommit = True

cur.execute(f"SELECT count(*), count(DISTINCT member_id), max(model_version), max(generated_at) "
            f"FROM {PG_SCHEMA}.{PG_TABLE}")
n_rows, n_members, mver, gen_at = cur.fetchone()
print(f"Lakebase {PG_SCHEMA}.{PG_TABLE}: {n_rows} rows, {n_members} members, "
      f"model v{mver}, generated {gen_at}; SELECT granted to app SP {APP_SP}")
```
Output:
```text
Read 500 rows / 100 members from bhavik_demos.offer_prioritization.member_offer_recommendations_with_reasoning
Lakebase offer_prioritization.lakebase_offers: 500 rows, 100 members, model v3, generated 2026-10-27T20:26:38.795599+00:00; SELECT granted to app SP f9bec3ef-421b-4a07-bf08-46c6df20c976
```

 ## Member-lookup latency (the app's hot path)

```python
member_ids = pdf.member_id.unique().tolist()
lookup = (f"SELECT offer_id, offer_name, rank, priority_score, llm_reasoning, shap_factors "
          f"FROM {PG_SCHEMA}.{PG_TABLE} WHERE member_id = %s ORDER BY rank")
for m in member_ids[:5]:  # warm-up
    cur.execute(lookup, (m,)); cur.fetchall()

lat = []
for i in range(200):
    m = member_ids[i % len(member_ids)]
    t0 = time.perf_counter()
    cur.execute(lookup, (m,))
    cur.fetchall()
    lat.append((time.perf_counter() - t0) * 1000)
lat = np.array(lat)
latency = {"n_queries": int(len(lat)), "p50_ms": round(float(np.percentile(lat, 50)), 2),
           "p95_ms": round(float(np.percentile(lat, 95)), 2), "p99_ms": round(float(np.percentile(lat, 99)), 2),
           "max_ms": round(float(lat.max()), 2)}
print("Member lookup latency (job -> Lakebase, round trip):", json.dumps(latency))

explain = psycopg.ClientCursor(conn)  # EXPLAIN needs client-side parameter binding
explain.execute(f"EXPLAIN {lookup}", (member_ids[0],))
print("\n".join(r[0] for r in explain.fetchall()))

cur.execute(lookup, (member_ids[0],))
print(f"\nSample app payload for {member_ids[0]}:")
for r in cur.fetchall():
    print(f"  #{r[2]} {r[1]:<38} {r[3]:6.2f}  {r[4][:90]}...")
cur.close(); conn.close()
```
Output:
```text
Member lookup latency (job -> Lakebase, round trip): {"n_queries": 200, "p50_ms": 3.12, "p95_ms": 7.21, "p99_ms": 10.26, "max_ms": 22.12}
Sort  (cost=11.47..11.47 rows=2 width=140)
  Sort Key: rank
  ->  Bitmap Heap Scan on lakebase_offers  (cost=4.16..11.46 rows=2 width=140)
        Recheck Cond: (member_id = 'MBR_014716CC'::text)
        ->  Bitmap Index Scan on idx_lakebase_offers_member  (cost=0.00..4.16 rows=2 width=0)
              Index Cond: (member_id = 'MBR_014716CC'::text)

Sample app payload for MBR_014716CC:
  #1 HSA/FSA Optimization                    80.24  This member is still paying a large share of costs toward their deductible, so learning ho...
  #2 Annual Wellness Visit Reminder          70.28  An Annual Wellness Visit Reminder is a good next step because this member hasn’t used thei...
  #3 Mental Health Support                   68.90  The Mental Health Support program can help this member build skills to manage everyday str...
  #4 Stress Management Workshop              61.61  The Stress Management Workshop is a good next step because it focuses on building everyday...
  #5 Respiratory Care Program                56.87  The Respiratory Care Program is a good next step because it offers education and support t...
```

```python
dbutils.notebook.exit(json.dumps({"rows": n_rows, "members": n_members, "model_version": mver,
                                  "latency": latency}))
```
Output:
```text
{"rows": 500, "members": 100, "model_version": "3", "latency": {"n_queries": 200, "p50_ms": 3.12, "p95_ms": 7.21, "p99_ms": 10.26, "max_ms": 22.12}}
```
