-- Silver: typed benefits utilization by benefit type for the current plan year.
CREATE OR REFRESH STREAMING TABLE silver_benefits_utilization (
  CONSTRAINT valid_member_id EXPECT (member_id IS NOT NULL) ON VIOLATION DROP ROW,
  CONSTRAINT valid_utilization_rate EXPECT (utilization_rate BETWEEN 0 AND 1) ON VIOLATION DROP ROW,
  CONSTRAINT used_within_max EXPECT (used_amount <= annual_max)
)
COMMENT 'Plan-year benefits utilization per member and benefit type (synthetic): annual max, used, remaining, utilization rate. Grain: member x benefit_type.'
TBLPROPERTIES ('quality' = 'silver')
AS SELECT
  member_id,
  benefit_type,
  CAST(annual_max AS DOUBLE) AS annual_max,
  CAST(used_amount AS DOUBLE) AS used_amount,
  CAST(remaining_balance AS DOUBLE) AS remaining_balance,
  CAST(utilization_rate AS DOUBLE) AS utilization_rate,
  CAST(last_used_date AS TIMESTAMP) AS last_used_date,
  CAST(claims_count AS INT) AS claims_count,
  CAST(plan_year AS INT) AS plan_year,
  batch_id,
  _ingested_at
FROM STREAM(bronze_benefits_utilization);
