-- Silver: typed, validated member demographics. Rows failing identity or range
-- checks are dropped and counted in the pipeline event log (data quality metrics).
CREATE OR REFRESH STREAMING TABLE silver_members (
  CONSTRAINT valid_member_id EXPECT (member_id IS NOT NULL AND member_id LIKE 'MBR_%') ON VIOLATION DROP ROW,
  CONSTRAINT valid_age EXPECT (age BETWEEN 0 AND 120) ON VIOLATION DROP ROW,
  CONSTRAINT valid_risk_score EXPECT (risk_score BETWEEN 0 AND 100),
  CONSTRAINT valid_plan_type EXPECT (plan_type IN ('HMO', 'PPO', 'EPO', 'HDHP', 'POS'))
)
COMMENT 'Health-plan member demographics and enrollment (synthetic), typed and validated. Grain: one row per member per extract batch.'
TBLPROPERTIES ('quality' = 'silver')
AS SELECT
  member_id,
  CAST(age AS INT) AS age,
  gender,
  region,
  plan_type,
  CAST(tenure_months AS INT) AS tenure_months,
  income_bracket,
  CAST(family_size AS INT) AS family_size,
  CAST(enrollment_date AS TIMESTAMP) AS enrollment_date,
  CAST(risk_score AS DOUBLE) AS risk_score,
  CAST(chronic_condition_count AS INT) AS chronic_condition_count,
  batch_id,
  _ingested_at
FROM STREAM(bronze_members);
