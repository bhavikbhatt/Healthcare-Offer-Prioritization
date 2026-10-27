# Platform evidence: bhavik_demos.offer_prioritization

Collected 2026-10-27T20:45:18+00:00 from workspace `fevm-classic-stable` with the Databricks CLI (`scripts/collect_platform_evidence.py`).

## 1. Unity Catalog: governed assets

Bronze and silver streaming tables accumulate every landed extract batch (Auto Loader is incremental:
each job run lands one new batch). Gold views, features and the model use the latest `batch_id` only.

| table_name | table_type | row_count | comment |
| --- | --- | --- | --- |
| bronze_benefits_utilization | STREAMING_TABLE | 105000 | Raw benefits_utilization extract (synthetic) ingested with Auto Loader from the raw_landing volume. One row pe |
| bronze_claims | STREAMING_TABLE | 64050 | Raw claims extract (synthetic) ingested with Auto Loader from the raw_landing volume. One row per source recor |
| bronze_engagement_history | STREAMING_TABLE | 189525 | Raw engagement_history extract (synthetic) ingested with Auto Loader from the raw_landing volume. One row per  |
| bronze_members | STREAMING_TABLE | 15000 | Raw members extract (synthetic) ingested with Auto Loader from the raw_landing volume. One row per source reco |
| gold_claims_monthly | MATERIALIZED_VIEW | 15735 | Monthly claims cost and utilization (synthetic) by region, plan type, claim type, diagnosis category and netwo |
| gold_engagement_performance | MATERIALIZED_VIEW | 5846 | Monthly member outreach performance (synthetic) by channel, engagement type, region and plan type: touches, re |
| gold_member_360 | MATERIALIZED_VIEW | 4980 | Member 360 profile (synthetic) for the latest extract batch: demographics, risk tier, 12-month claims cost and |
| member_features | MANAGED | 4980 | 80+ engineered member features (demographics, claims, diagnosis flags, benefits, engagement, temporal) used to |
| member_offer_recommendations_with_reasoning | MANAGED | 500 | Care-manager work queue: top-5 offers for sampled members with SHAP decision factors and LLM-written plain-lan |
| model_feature_importance | MANAGED | 96 | Mean LightGBM gain per feature across the 16 per-offer models for the current champion model version. |
| offer_catalog | MANAGED | 16 | Catalog of the 16 care and wellness programs the plan can offer (preventive, chronic disease, mental health, p |
| offer_feedback | MANAGED | 2 | Care-manager approve/reject decisions and free-text comments captured in the Offer Management Console. Ground  |
| offer_recommendations | MANAGED | 24900 | Model output: top-5 next-best offers for every member with priority score (0-100), rank and model version. Gra |
| silver_benefits_utilization | STREAMING_TABLE | 104860 | Plan-year benefits utilization per member and benefit type (synthetic): annual max, used, remaining, utilizati |
| silver_claims | STREAMING_TABLE | 63794 | Medical claims (synthetic) with claim type, diagnosis category, billed/allowed/paid amounts and network status |
| silver_engagement_history | STREAMING_TABLE | 189145 | Member engagement events (synthetic) across email, app, phone, mail, portal and SMS, with response flag. Grain |
| silver_members | STREAMING_TABLE | 14960 | Health-plan member demographics and enrollment (synthetic), typed and validated. Grain: one row per member per |

### Governed tags

| level | object | tag_name | tag_value |
| --- | --- | --- | --- |
| table | bronze_benefits_utilization | Domain | Healthcare |
| table | bronze_benefits_utilization | data_classification | pii |
| table | bronze_benefits_utilization | medallion_layer | bronze |
| table | bronze_claims | Domain | Healthcare |
| table | bronze_claims | data_classification | pii |
| table | bronze_claims | medallion_layer | bronze |
| table | bronze_engagement_history | Domain | Healthcare |
| table | bronze_engagement_history | data_classification | pii |
| table | bronze_engagement_history | medallion_layer | bronze |
| table | bronze_members | Domain | Healthcare |
| table | bronze_members | data_classification | pii |
| table | bronze_members | medallion_layer | bronze |
| table | gold_claims_monthly | Domain | Healthcare |
| table | gold_claims_monthly | medallion_layer | gold |
| table | gold_engagement_performance | Domain | Healthcare |
| table | gold_engagement_performance | medallion_layer | gold |
| table | gold_member_360 | Domain | Healthcare |
| table | gold_member_360 | data_classification | pii |
| table | gold_member_360 | medallion_layer | gold |
| table | member_features | Domain | Healthcare |
| table | member_features | data_classification | pii |
| table | member_features | medallion_layer | gold |
| table | member_offer_recommendations_with_reasoning | Domain | Healthcare |
| table | member_offer_recommendations_with_reasoning | data_classification | pii |
| table | member_offer_recommendations_with_reasoning | medallion_layer | gold |
| table | model_feature_importance | Domain | Healthcare |
| table | model_feature_importance | medallion_layer | gold |
| table | offer_catalog | Domain | Healthcare |
| table | offer_catalog | medallion_layer | gold |
| table | offer_feedback | Domain | Healthcare |
| table | offer_feedback | data_classification | pii |
| table | offer_feedback | medallion_layer | gold |
| table | offer_recommendations | Domain | Healthcare |
| table | offer_recommendations | data_classification | pii |
| table | offer_recommendations | medallion_layer | gold |
| table | silver_benefits_utilization | Domain | Healthcare |
| table | silver_benefits_utilization | data_classification | pii |
| table | silver_benefits_utilization | medallion_layer | silver |
| table | silver_claims | Domain | Healthcare |
| table | silver_claims | data_classification | pii |
| table | silver_claims | medallion_layer | silver |
| table | silver_engagement_history | Domain | Healthcare |
| table | silver_engagement_history | data_classification | pii |
| table | silver_engagement_history | medallion_layer | silver |
| table | silver_members | Domain | Healthcare |
| table | silver_members | data_classification | pii |
| table | silver_members | medallion_layer | silver |
| column | member_features.age | phi | true |
| column | member_features.member_id | phi | health_plan_beneficiary_number |
| column | member_features.risk_score | phi | true |
| column | member_offer_recommendations_with_reasoning.age | phi | true |
| column | member_offer_recommendations_with_reasoning.has_cardiovascular | phi | true |
| column | member_offer_recommendations_with_reasoning.has_diabetes | phi | true |
| column | member_offer_recommendations_with_reasoning.has_mental_health | phi | true |
| column | member_offer_recommendations_with_reasoning.has_respiratory | phi | true |
| column | member_offer_recommendations_with_reasoning.member_id | phi | health_plan_beneficiary_number |
| column | member_offer_recommendations_with_reasoning.risk_score | phi | true |
| column | offer_feedback.feedback_text | phi | true |
| column | offer_feedback.member_id | phi | health_plan_beneficiary_number |
| column | offer_recommendations.member_id | phi | health_plan_beneficiary_number |

### Lineage (system.access.table_lineage)

Pipeline hops are table-to-table. The ML notebooks read silver/features into pandas, so UC records those
steps as job reads and job writes rather than a direct edge.

| source | target | entity_type | events | last_seen |
| --- | --- | --- | --- | --- |
| (external / file) | bronze_benefits_utilization | PIPELINE | 4 | 2026-10-27T20:20:23.245Z |
| (external / file) | bronze_claims | PIPELINE | 4 | 2026-10-27T20:20:23.249Z |
| (external / file) | bronze_engagement_history | PIPELINE | 5 | 2026-10-27T20:20:23.265Z |
| (external / file) | bronze_members | PIPELINE | 4 | 2026-10-27T20:20:23.234Z |
| bronze_benefits_utilization | silver_benefits_utilization | PIPELINE | 4 | 2026-10-27T20:20:29.570Z |
| bronze_claims | silver_claims | PIPELINE | 5 | 2026-10-27T20:20:28.436Z |
| bronze_engagement_history | silver_engagement_history | PIPELINE | 4 | 2026-10-27T20:20:28.813Z |
| bronze_members | silver_members | PIPELINE | 4 | 2026-10-27T20:20:28.010Z |
| silver_benefits_utilization | gold_member_360 | PIPELINE | 3 | 2026-10-27T20:20:53.697Z |
| silver_claims | (read by job) | PIPELINE | 4 | 2026-10-27T20:20:42.223Z |
| silver_claims | gold_claims_monthly | PIPELINE | 3 | 2026-10-27T20:20:51.617Z |
| silver_claims | gold_member_360 | PIPELINE | 3 | 2026-10-27T20:20:53.697Z |
| silver_engagement_history | (read by job) | PIPELINE | 3 | 2026-10-27T20:20:42.217Z |
| silver_engagement_history | gold_engagement_performance | PIPELINE | 3 | 2026-10-27T20:20:51.632Z |
| silver_engagement_history | gold_member_360 | PIPELINE | 3 | 2026-10-27T20:20:53.697Z |
| silver_members | (read by job) | PIPELINE | 11 | 2026-10-27T20:20:43.421Z |
| silver_members | gold_claims_monthly | PIPELINE | 3 | 2026-10-27T20:20:51.617Z |
| silver_members | gold_engagement_performance | PIPELINE | 3 | 2026-10-27T20:20:51.632Z |
| silver_members | gold_member_360 | PIPELINE | 3 | 2026-10-27T20:20:53.697Z |
| (external / file) | member_features | JOB | 3 | 2026-10-27T20:22:25.523Z |
| (external / file) | member_offer_recommendations_with_reasoning | JOB | 3 | 2026-10-27T20:26:42.783Z |
| (external / file) | model_feature_importance | JOB | 3 | 2026-10-27T20:23:21.831Z |
| (external / file) | offer_catalog | JOB | 3 | 2026-10-27T20:22:28.095Z |
| (external / file) | offer_feedback | JOB | 4 | 2026-10-27T20:27:33.205Z |
| (external / file) | offer_recommendations | JOB | 3 | 2026-10-27T20:24:21.027Z |
| member_features | (read by job) | JOB | 3 | 2026-10-27T20:24:13.486Z |
| member_offer_recommendations_with_reasoning | (read by job) | JOB | 6 | 2026-10-27T20:27:16.434Z |
| silver_benefits_utilization | (read by job) | JOB | 3 | 2026-10-27T20:22:14.507Z |
| silver_claims | (read by job) | JOB | 3 | 2026-10-27T20:22:13.246Z |
| silver_engagement_history | (read by job) | JOB | 3 | 2026-10-27T20:22:15.583Z |
| silver_members | (read by job) | JOB | 6 | 2026-10-27T20:22:12.117Z |
| gold_claims_monthly | (read by job) |  | 8 | 2026-10-27T20:29:24.669Z |
| gold_engagement_performance | (read by job) |  | 8 | 2026-10-27T20:28:51.410Z |
| gold_member_360 | (read by job) |  | 10 | 2026-10-27T20:29:11.574Z |
| member_offer_recommendations_with_reasoning | (read by job) |  | 5 | 2026-10-27T20:29:43.063Z |
| offer_catalog | (read by job) |  | 8 | 2026-10-27T20:28:34.978Z |
| offer_feedback | (read by job) |  | 2 | 2026-10-27T20:28:33.177Z |
| offer_recommendations | (read by job) |  | 8 | 2026-10-27T20:29:11.574Z |

## 2. Model registry (Unity Catalog)

`bhavik_demos.offer_prioritization.healthcare_offer_prioritizer`: `@champion` -> **v3** (run `4c2ee4573c4f4a7e990c704ddb057f4e`)

| version | status | created | comment |
| --- | --- | --- | --- |
| 1 | READY | 2026-10-27T19:58:43+00:00 | Trained on extract batch 20261027T195456Z; test R2=0.778, top-5 overlap=0.852 |
| 2 | READY | 2026-10-27T20:10:12+00:00 | Trained on extract batch 20261027T200615Z; test R2=0.782, top-5 overlap=0.850 |
| 3 | READY | 2026-10-27T20:23:16+00:00 | Trained on extract batch 20261027T201919Z; test R2=0.784, top-5 overlap=0.846 |

## 3. Model output

### offer_recommendations by rank

| rank | recommendations | avg_score | min_score | max_score |
| --- | --- | --- | --- | --- |
| 1 | 4980 | 91.4 | 66.57 | 100.0 |
| 2 | 4980 | 83.0 | 61.33 | 98.48 |
| 3 | 4980 | 76.8 | 54.98 | 97.43 |
| 4 | 4980 | 72.1 | 53.94 | 94.63 |
| 5 | 4980 | 68.7 | 51.79 | 92.82 |

### Work-queue sample: highest-risk member, with LLM reasoning (databricks-gpt-5-1)

**#1 Heart Health Program** (score 98.69, source: llm)

> The Heart Health Program is a strong next step because this member is living with a heart-related condition and may benefit from extra support in managing it day to day. At age 77, tailored guidance around medications, symptoms to watch for, and healthy routines can help them stay as active and independent as possible. Their higher risk level means having a dedicated team to check in, answer questions, and coordinate care could help catch issues early and prevent complications. Over their many years with the plan, this added layer of support can build on the care they already receive and help them feel more confident about their heart health.

**#2 Cancer Screening Program** (score 89.12, source: llm)

> The Cancer Screening Program is a good next step because at age 77, staying up to date on screenings can help catch any issues early, when they’re often easier to treat. With your health history and higher risk level, having a team review which cancer screenings are most appropriate for you adds an extra layer of protection. The program can also help coordinate appointments and follow-up, making it easier to stay on top of your health.

**#3 Annual Wellness Visit Reminder** (score 71.14, source: llm)

> This member is 77, so staying ahead with preventive care is especially important to help maintain independence and quality of life. An Annual Wellness Visit gives the doctor time to review heart health, medications, and any subtle changes that might not come up in a regular sick visit. Even though they haven’t had many recent claims, this visit can help catch issues early and update a care plan tailored to their needs. It’s a simple way to make sure they’re getting the right support for their more complex health needs as they age.

**#4 Care Navigator Assignment** (score 70.0, source: llm)

> The Care Navigator Assignment program can help this member better coordinate care around their heart-related condition and other recent health needs. With several different diagnoses and more complex care needs, it can be hard to keep track of appointments, tests, and follow-ups. A care navigator can work alongside the member and their doctors to simplify next steps, answer questions, and help them get the most out of their benefits.

**#5 HSA/FSA Optimization** (score 66.77, source: llm)

> The HSA/FSA Optimization program can help this member make the most of every dollar they set aside for health expenses, especially given how often they use their medical benefits compared with the average member. At age 77 and with more complex care needs, they may have ongoing costs like specialist visits, tests, or therapies that are easier to manage with a clear plan for pre-tax spending. This program can walk them through what’s eligible, how to plan for upcoming expenses, and how to avoid leaving any available funds unused.

## 4. Databricks App + Lakebase

| app | url | app_status | compute | active_deployment | deployed_at | resources |
| --- | --- | --- | --- | --- | --- | --- |
| offer-prioritization-console | https://offer-prioritization-console-7474656617733963.aws.databricksapps.com | RUNNING | ACTIVE | 01f1d245ee021a4e830c4daad9701e2b | 2026-10-27T20:35:27Z | postgres=postgres |

| endpoint | host | state |
| --- | --- | --- |
| projects/offer-prioritization/branches/production/endpoints/primary | ep-sweet-shadow-d2du2zdd.database.us-east-1.cloud.databricks.com | ACTIVE |

## 5. Genie Agent

| title | space_id | warehouse_id | description |
| --- | --- | --- | --- |
| Member Offer Intelligence | 01f1d2431e1c1007bbd755dc7cc91fbb | c5373a0bb5c8d31b | Ask questions about health-plan members, risk, claims cost, outreach response and the AI next-best-offer recommendations |

Question-and-answer transcript with generated SQL: [notebooks/genie_agent.md](notebooks/genie_agent.md).
