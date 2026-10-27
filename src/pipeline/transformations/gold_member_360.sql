-- Gold: one row per member for the latest extract batch. A curated member
-- profile used by the Genie Agent, care managers and the offer model.
CREATE OR REFRESH MATERIALIZED VIEW gold_member_360
COMMENT 'Member 360 profile (synthetic) for the latest extract batch: demographics, risk tier, 12-month claims cost and utilization, condition flags, engagement responsiveness and benefits utilization. Grain: one row per member.'
TBLPROPERTIES ('quality' = 'gold')
AS
WITH latest AS (
  SELECT max(batch_id) AS batch_id FROM silver_members
),
m AS (
  SELECT s.* FROM silver_members s JOIN latest l ON s.batch_id = l.batch_id
),
c AS (
  SELECT
    s.member_id,
    count(*) AS claims_12m,
    round(sum(s.paid_amount), 2) AS paid_amount_12m,
    round(sum(s.member_responsibility), 2) AS member_cost_12m,
    sum(CASE WHEN s.claim_type = 'emergency' THEN 1 ELSE 0 END) AS er_visits_12m,
    sum(CASE WHEN s.claim_type = 'inpatient' THEN 1 ELSE 0 END) AS inpatient_admits_12m,
    sum(CASE WHEN s.claim_type = 'preventive' THEN 1 ELSE 0 END) AS preventive_visits_12m,
    max(CASE WHEN s.diagnosis_category = 'diabetes' THEN 1 ELSE 0 END) AS has_diabetes,
    max(CASE WHEN s.diagnosis_category = 'cardiovascular' THEN 1 ELSE 0 END) AS has_cardiovascular,
    max(CASE WHEN s.diagnosis_category = 'respiratory' THEN 1 ELSE 0 END) AS has_respiratory,
    max(CASE WHEN s.diagnosis_category = 'mental_health' THEN 1 ELSE 0 END) AS has_mental_health,
    max(CASE WHEN s.diagnosis_category = 'musculoskeletal' THEN 1 ELSE 0 END) AS has_musculoskeletal,
    count(DISTINCT s.diagnosis_category) AS distinct_diagnosis_categories,
    round(avg(CASE WHEN s.in_network THEN 1.0 ELSE 0.0 END), 3) AS in_network_rate,
    max(s.claim_date) AS last_claim_date
  FROM silver_claims s JOIN latest l ON s.batch_id = l.batch_id
  WHERE s.claim_date >= current_timestamp() - INTERVAL 365 DAYS
  GROUP BY s.member_id
),
e AS (
  SELECT
    s.member_id,
    count(*) AS engagements_12m,
    round(avg(CASE WHEN s.response_flag THEN 1.0 ELSE 0.0 END), 3) AS engagement_response_rate,
    mode(s.channel) AS most_used_channel,
    max(s.engagement_date) AS last_engagement_date
  FROM silver_engagement_history s JOIN latest l ON s.batch_id = l.batch_id
  GROUP BY s.member_id
),
b AS (
  SELECT
    s.member_id,
    round(avg(s.utilization_rate), 3) AS avg_benefit_utilization,
    round(max(CASE WHEN s.benefit_type = 'preventive' THEN s.utilization_rate END), 3) AS preventive_benefit_utilization
  FROM silver_benefits_utilization s JOIN latest l ON s.batch_id = l.batch_id
  GROUP BY s.member_id
)
SELECT
  m.member_id,
  m.age,
  CASE WHEN m.age < 30 THEN '18-29' WHEN m.age < 45 THEN '30-44'
       WHEN m.age < 65 THEN '45-64' ELSE '65+' END AS age_band,
  m.gender,
  m.region,
  m.plan_type,
  m.income_bracket,
  m.family_size,
  m.tenure_months,
  round(m.risk_score, 1) AS risk_score,
  CASE WHEN m.risk_score >= 60 THEN 'High' WHEN m.risk_score >= 30 THEN 'Medium' ELSE 'Low' END AS risk_tier,
  m.chronic_condition_count,
  coalesce(c.claims_12m, 0) AS claims_12m,
  coalesce(c.paid_amount_12m, 0) AS paid_amount_12m,
  coalesce(c.member_cost_12m, 0) AS member_cost_12m,
  coalesce(c.er_visits_12m, 0) AS er_visits_12m,
  coalesce(c.inpatient_admits_12m, 0) AS inpatient_admits_12m,
  coalesce(c.preventive_visits_12m, 0) AS preventive_visits_12m,
  coalesce(c.has_diabetes, 0) AS has_diabetes,
  coalesce(c.has_cardiovascular, 0) AS has_cardiovascular,
  coalesce(c.has_respiratory, 0) AS has_respiratory,
  coalesce(c.has_mental_health, 0) AS has_mental_health,
  coalesce(c.has_musculoskeletal, 0) AS has_musculoskeletal,
  coalesce(c.distinct_diagnosis_categories, 0) AS distinct_diagnosis_categories,
  c.in_network_rate,
  c.last_claim_date,
  coalesce(e.engagements_12m, 0) AS engagements_12m,
  e.engagement_response_rate,
  e.most_used_channel,
  e.last_engagement_date,
  b.avg_benefit_utilization,
  b.preventive_benefit_utilization,
  m.batch_id
FROM m
LEFT JOIN c ON m.member_id = c.member_id
LEFT JOIN e ON m.member_id = e.member_id
LEFT JOIN b ON m.member_id = b.member_id;
