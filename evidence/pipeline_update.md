# Lakeflow pipeline update `ab4bc1f6-725c-4775-8e49-767c5f79203e`

Pipeline `af3f71ac-5d82-492d-8f38-cd37284e7136` (offer-prioritization-medallion). Queried from the pipeline event log with
`SELECT ... FROM event_log('af3f71ac-5d82-492d-8f38-cd37284e7136')`.

## Rows written per flow (bronze -> silver -> gold)

| flow | rows_written | rows_dropped |
| --- | --- | --- |
| bronze_benefits_utilization | 35000 | 0 |
| bronze_claims | 21350 | 0 |
| bronze_engagement_history | 63175 | 0 |
| bronze_members | 5000 | 0 |
| gold_claims_monthly | 15735 | 0 |
| gold_engagement_performance | 5846 | 0 |
| gold_member_360 | 4980 | 0 |
| silver_benefits_utilization | 34930 | 70 |
| silver_claims | 21222 | 128 |
| silver_engagement_history | 62985 | 190 |
| silver_members | 4980 | 20 |

## Data-quality expectations

Expectations declared `ON VIOLATION DROP ROW` remove the row; the others keep it and record the violation.

| flow | expectation | passed | failed |
| --- | --- | --- | --- |
| silver_benefits_utilization | used_within_max | 35000 | 0 |
| silver_benefits_utilization | valid_member_id | 35000 | 0 |
| silver_benefits_utilization | valid_utilization_rate | 34930 | 70 |
| silver_claims | non_negative_amount | 21265 | 85 |
| silver_claims | paid_not_above_billed | 21201 | 149 |
| silver_claims | valid_claim_date | 21350 | 0 |
| silver_claims | valid_claim_id | 21350 | 0 |
| silver_claims | valid_member_id | 21307 | 43 |
| silver_engagement_history | valid_channel | 62985 | 190 |
| silver_engagement_history | valid_engagement_id | 63175 | 0 |
| silver_engagement_history | valid_member_id | 63175 | 0 |
| silver_members | valid_age | 4980 | 20 |
| silver_members | valid_member_id | 5000 | 0 |
| silver_members | valid_plan_type | 4990 | 10 |
| silver_members | valid_risk_score | 5000 | 0 |

## Update timeline

| timestamp | level | message |
| --- | --- | --- |
| 2026-10-27T20:19:37.852Z | INFO | Update ab4bc1 started by JOB_TASK. |
| 2026-10-27T20:19:40.440Z | INFO | Update ab4bc1 is WAITING_FOR_RESOURCES. |
| 2026-10-27T20:19:51.642Z | INFO | Update ab4bc1 is INITIALIZING. |
| 2026-10-27T20:20:00.244Z | INFO | Update ab4bc1 is SETTING_UP_TABLES. |
| 2026-10-27T20:20:02.363Z | INFO | Update ab4bc1 is RUNNING. |
| 2026-10-27T20:20:53.723Z | INFO | Update ab4bc1 is COMPLETED. |
