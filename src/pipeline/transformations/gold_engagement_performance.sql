-- Gold: outreach performance by channel and engagement type — the baseline
-- the offer model is trying to beat (generic outreach response rates).
CREATE OR REFRESH MATERIALIZED VIEW gold_engagement_performance
COMMENT 'Monthly member outreach performance (synthetic) by channel, engagement type, region and plan type: touches, responders and response rate, for the latest extract batch. Grain: month x channel x engagement_type x region x plan_type.'
TBLPROPERTIES ('quality' = 'gold')
AS
WITH latest AS (SELECT max(batch_id) AS batch_id FROM silver_members)
SELECT
  date_trunc('MONTH', e.engagement_date) AS engagement_month,
  e.channel,
  e.engagement_type,
  m.region,
  m.plan_type,
  count(*) AS touches,
  count(DISTINCT e.member_id) AS members_touched,
  sum(CASE WHEN e.response_flag THEN 1 ELSE 0 END) AS responses,
  round(avg(CASE WHEN e.response_flag THEN 1.0 ELSE 0.0 END), 4) AS response_rate
FROM silver_engagement_history e
JOIN latest l ON e.batch_id = l.batch_id
JOIN silver_members m ON e.member_id = m.member_id AND m.batch_id = l.batch_id
GROUP BY ALL;
