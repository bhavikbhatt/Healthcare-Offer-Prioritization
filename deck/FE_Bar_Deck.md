# HLS - Personalized Offer Recommendations (FE Bar business deck)

Text export of the presentation for the FE Bar evaluator. Slides, tables, speaker notes and a text
description of every image or video slide are included.

## Slide 1: HLS - Personalized Offer Recommendations

Bhavik Bhatt

## Slide 2: The Right Program for Every Member

- The outcome
  - Lift care-program enrollment from 8% to 11% of eligible members, worth ~$5.8M a year for a 1M-member plan (illustrative value model).
- How
  - One AI ranking of 16 care and wellness programs for every member, with a plain-language "why" a care manager can say out loud, served in the console they already work in.
- Proof it works (this build, synthetic data)
  - 4,980 members scored on 96 features in one governed Databricks pipeline; 100 member work queue served from Lakebase at p95 7.2 ms; 5/5 business questions answered by the Genie Agent.
- The ask
  - A 90-day pilot with one care-management team: control vs. AI-ranked outreach, decision gate on enrollment lift and minutes saved per case.

> Speaker notes: Purpose: lead with the business outcome before any architecture. Talk track: enrollment lift and care-manager time are the two numbers the sponsor and the domain owner each own. Evidence: value model on slide 6 (assumptions stated); platform proof points on slide 9 come from the committed job run. Transition: why generic outreach under-performs today.

## Slide 3: Overcoming the Limitations of Generic Care

- The Engagement Gap:
  - Traditional rule-based engines rely on static logic (e.g., "If Age > 65 and Diabetic, then offer Program X"), which often misses the nuance of a member’s current health journey, leading to low offer acceptance.
- Operational Silos:
  - Care managers often have to navigate multiple systems to find relevant programs, creating friction during time-sensitive member interactions.
- Information Overload:
  - With dozens of available wellness programs (Heart Health, Weight Management, etc.), manual selection is inefficient and leads to "one-size-fits-all" fatigue for the member.
- The Explainability Gap:
  - Traditional models act as a "black box," providing recommendations without the "why," which makes it difficult for care managers to build trust with members.

## Slide 4: Enabling Intelligent, Personalized Care Workflows

- The Proposed Solution:
  - An end-to-end AI recommendation system built on the Databricks Data Intelligence Platform. It replaces rigid, rule-based workflows with model-driven "Next Best Action" offers, utilizing Generative AI to provide human-readable reasoning for every recommendation.

## Slide 5: Healthcare Outcomes

- Elevating Member Engagement (Personalize Engagement)
  - By moving from generic outreach to hyper-personalized "Next Best Action" offers, the system increases the Propensity to Engage with social determinants and wellness programs.
- Proactive Risk Mitigation (Predict Disease Risk)
  - AI identifies sub-clinical trends in the "Gold" data layer to recommend Heart Health or Preventative programs before acute events occur.
- Maximizing Member Retention (Improve Retention)
  - Relevant, timely wellness offers make members feel supported rather than "processed." This personalized care model reduces Member Churn by addressing individual health goals.

## Slide 6: Quantified Impact: ~$5.8M a Year


| KPI (owner) | Baseline | Pilot target | How the value is calculated | Annual value |
| --- | --- | --- | --- | --- |
| Care-program enrollment (Care Mgmt / Exec sponsor) | 8% of 200K eligible | 11% (+3 pts) | +6,000 enrollees x $600 avoided medical cost per enrollee per year | $3.6M |
| Care-manager productivity (Dir. Care Mgmt) | ~25 min prep / case | -10 min / case | 60 care managers x 40 cases/wk x 46 wks x 10 min = 18,400 hrs x $55/hr | $1.0M |
| Voluntary churn (Exec sponsor) | 12.0% | 11.8% (-0.2 pts) | 2,000 members retained x $350 annual margin per member | $0.7M |
| Outreach waste (Marketing ops) | 3M generic touches / yr | -25% irrelevant touches | 750K fewer touches x $0.60 blended cost per touch | $0.45M |
| Total |  |  | Excludes Stars/HEDIS quality-bonus upside from preventive-screening gap closure | ~$5.8M |

Illustrative value model for a 1M-member health plan; assumptions are deliberately conservative and are replaced with the plan's own baselines in discovery. Lift is proven only by the pilot's control group. The prototype runs on synthetic data, so model metrics measure fit to clinical targeting rules, not observed acceptance.

> Speaker notes: Purpose: quantify value in the buyer's KPIs. Each row names the KPI owner, the baseline, the pilot target and the math. Assumptions are deliberately conservative and should be replaced with the plan's own baselines in discovery. Likely question: how do you know enrollment will lift? Answer: we don't yet; the pilot's control group measures it, and the decision gate is on that number.

## Slide 7: What Each Stakeholder Gets

- Executive sponsor: VP Member Engagement / Chief Growth Officer
  - KPIs: program enrollment, voluntary churn, cost of care (PMPY), Stars/HEDIS gap closure.
  - Value: more members in the programs that bend cost; less spend on outreach members ignore.
  - Risk control: every recommendation is governed, explainable and auditable in Unity Catalog.
- Domain owner: Director of Care Management
  - KPIs: members engaged per care manager, prep minutes per case, program acceptance rate.
  - Value: a ranked work queue with the "why" written for the call; one screen instead of several systems.
  - Control: approve or reject any offer; feedback retrains the model (champion/challenger).

> Speaker notes: Purpose: frame value separately for the exec sponsor and the domain owner. The sponsor buys enrollment and retention; the care-management director buys time back and trust in the recommendation.

## Slide 8: Architecture & Demo


## Slide 9: What Ran: One Integrated Journey


| Stage | What it does | Proof from the committed job run (synthetic data) |
| --- | --- | --- |
| Lakeflow | Auto Loader bronze, silver with 15 data-quality expectations, 3 gold views | 125K raw records ingested; exactly the 408 injected bad rows dropped, 159 flagged |
| Unity Catalog | Comments, governed tags (Domain, phi, pii), least-privilege grants, lineage, model registry | 63 governance statements applied, 0 skipped; member_id tagged as HIPAA beneficiary number |
| ML + Gen AI | LightGBM x16 offers, MLflow, UC @champion; SHAP drivers; GPT-5.1 writes the why | Test R2 0.78, top-5 overlap 0.85; 500 explanations, 100% LLM, 0% jargon |
| Lakebase | Care-manager work queue served from managed Postgres | Member lookup p50 3.1 ms, p95 7.2 ms (200 queries) |
| Genie Agent | Member Offer Intelligence over 7 governed tables | 5/5 business questions answered with generated SQL |
| Databricks App | Offer Management Console: lookup, top 5 with reasons, approve/reject | Live test PASS: approval and comment landed in offer_feedback (UC) |


> Speaker notes: Purpose: show the six platform stages are one connected journey, each with execution proof from the committed run (evidence/ folder in the repo). Transition: walk the lifecycle in more detail.

## Slide 10: End-to-End Recommendation Lifecycle

- Data & Feature Engineering:
  - Lakeflow Declarative Pipeline ingests raw member, claims, benefits and engagement extracts with Auto Loader: bronze, silver (data-quality expectations), gold.
- Model Training & Tracking:
  - Multi-output LightGBM scores all 16 offers per member; tracked in MLflow and registered in Unity Catalog with a champion alias.
- Governance:
  - Centralized management of all data and AI assets (models, features, functions) through Unity Catalog.
- Batch Inference:
  - Generate the top five recommendations for thousands of members simultaneously, stored in Delta inference tables.

## Slide 11: From Scoring to Reasoning: Explainable AI

- Feature Importance:
  - Utilize Shapley values to identify which of the 96 member features (e.g., recent claims, engagement history) drove a specific recommendation.
- Generative AI Integration:
  - The Databricks AI Gateway provides secure access to LLMs like GPT 5.1.
- Reasoning Generation:
  - A specialized prompt combines member profiles and Shapley factors, allowing the LLM to generate clear, textual justifications for the care manager.

## Slide 12: The Care Manager Experience

- Real-time Insights:
  - A custom frontend hosted on Databricks Apps pulls high-level member profiles from Databricks Lakebase (managed PostgreSQL).
- Actionable Data:
  - Displays relevance scores and AI-generated reasoning to support clinical decision-making.
- Active Feedback Loop:
  - Campaign managers can approve or reject offers with notes. This feedback is saved to Delta tables and used for scheduled automated retraining.
- Model Promotion:
  - Retraining creates "Challenger" versions in Unity Catalog, which are promoted to "Champion" based on MLflow evaluation metrics.

## Slide 13: (no title)

*[Architecture diagram (image). Inside Databricks: Bronze -> Silver -> Gold (Lakeflow) feed Features (MLflow); Training/Tuning/Evaluation (MLflow) and the Recommendation Engine call the AI Gateway, which reaches external foundation models (GPT-5) and the AutoML/scratch model; results go to Inference Tables, then Lakebase; the Offer Management Console (Databricks Apps) reads Lakebase, serves care managers and downstream modalities (call center, OCM), and sends Feedback back into training. Unity Catalog governance and observability span the platform on top of the cloud data lake.]*

## Slide 14: (no title)

*[Demo video of the Offer Management Console: member lookup (MBR_0A1ACD96, age 77, diabetes and complex care) loads from Lakebase; the #1 offer is the Diabetes Management Program with relevance 90 and an AI recommendation insight explaining why, with a 'View Decision Factors' button for the SHAP drivers.]*

## Slide 15: What the Demo Shows

- Care manager view (Offer Management Console, Databricks App)
  - Look up a member: age, risk score, conditions, tenure, claims and engagements load from Lakebase.
  - Top 5 offers ranked by relevance, each with an AI-written reason and "View Decision Factors" (SHAP).
- Act on it
  - Approve or reject an offer with a note; it lands in the governed offer_feedback Delta table.
- Ask the data (Genie Agent)
  - "Which offers are recommended most often as the #1 offer?" and "What is the outreach response rate by channel?" answered in plain English with the SQL shown.

## Slide 16: Next Steps: 90-Day Pilot

- Weeks 0-3: Connect
  - Point the Lakeflow pipeline at real eligibility, claims and CRM extracts; apply UC row filters and masks to PHI; agree on baselines.
- Weeks 4-9: Run
  - Randomize one care-management team's queue: AI-ranked vs. current rules (control). Capture approve/reject feedback and outcomes daily.
- Weeks 10-12: Decide
  - Gate: >=2 pt enrollment lift and >=10 min saved per case vs. control. Retrain on observed outcomes and promote challenger to champion only if it beats it.

## Slide 17: Resources


## Slide 18: (no title)

- [Databricks App - Offer Management Console](https://offer-prioritization-console-7474656617733963.aws.databricksapps.com/)
- [Github Repository](https://github.com/bhavikbhatt/Healthcare-Offer-Prioritization)
- [Demo Video](https://drive.google.com/file/d/1M6M_gos77SAcQrdcalGH-IeK5uXu42pi/view?usp=drive_link)
- [Architecture Diagram](https://drive.google.com/file/d/13-FexbI5Q_pzx3GgXRyEQyTM0EDDXhLv/view?usp=sharing)
- [Genie Agent - Member Offer Intelligence](https://fevm-classic-stable-tp909z.cloud.databricks.com/genie/rooms/01f1d2431e1c1007bbd755dc7cc91fbb)
- [Execution evidence (repo /evidence)](https://github.com/bhavikbhatt/Healthcare-Offer-Prioritization/tree/main/evidence)

## Slide 19: (no title)

