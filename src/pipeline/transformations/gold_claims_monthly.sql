-- Gold: monthly claims cost and utilization, keeping the dimensions analysts
-- slice by (region, plan type, claim type, diagnosis category, network status).
CREATE OR REFRESH MATERIALIZED VIEW gold_claims_monthly
COMMENT 'Monthly claims cost and utilization (synthetic) by region, plan type, claim type, diagnosis category and network status, for the latest extract batch. Grain: month x region x plan_type x claim_type x diagnosis_category x in_network.'
TBLPROPERTIES ('quality' = 'gold')
AS
WITH latest AS (SELECT max(batch_id) AS batch_id FROM silver_members)
SELECT
  date_trunc('MONTH', c.claim_date) AS claim_month,
  m.region,
  m.plan_type,
  c.claim_type,
  c.diagnosis_category,
  c.in_network,
  count(*) AS claim_count,
  count(DISTINCT c.member_id) AS member_count,
  round(sum(c.claim_amount), 2) AS billed_amount,
  round(sum(c.allowed_amount), 2) AS allowed_amount,
  round(sum(c.paid_amount), 2) AS paid_amount,
  round(sum(c.member_responsibility), 2) AS member_cost
FROM silver_claims c
JOIN latest l ON c.batch_id = l.batch_id
JOIN silver_members m ON c.member_id = m.member_id AND m.batch_id = l.batch_id
GROUP BY ALL;
