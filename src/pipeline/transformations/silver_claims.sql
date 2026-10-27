-- Silver: typed, validated medical claims.
CREATE OR REFRESH STREAMING TABLE silver_claims (
  CONSTRAINT valid_claim_id EXPECT (claim_id IS NOT NULL) ON VIOLATION DROP ROW,
  CONSTRAINT valid_member_id EXPECT (member_id IS NOT NULL) ON VIOLATION DROP ROW,
  CONSTRAINT valid_claim_date EXPECT (claim_date IS NOT NULL AND claim_date <= current_timestamp()) ON VIOLATION DROP ROW,
  CONSTRAINT non_negative_amount EXPECT (claim_amount >= 0) ON VIOLATION DROP ROW,
  CONSTRAINT paid_not_above_billed EXPECT (paid_amount <= claim_amount)
)
COMMENT 'Medical claims (synthetic) with claim type, diagnosis category, billed/allowed/paid amounts and network status. Grain: one row per claim.'
TBLPROPERTIES ('quality' = 'silver')
AS SELECT
  claim_id,
  member_id,
  CAST(claim_date AS TIMESTAMP) AS claim_date,
  claim_type,
  diagnosis_category,
  CAST(claim_amount AS DOUBLE) AS claim_amount,
  CAST(allowed_amount AS DOUBLE) AS allowed_amount,
  CAST(paid_amount AS DOUBLE) AS paid_amount,
  CAST(member_responsibility AS DOUBLE) AS member_responsibility,
  provider_type,
  CAST(in_network AS BOOLEAN) AS in_network,
  status,
  batch_id,
  _ingested_at
FROM STREAM(bronze_claims);
