# FE Bar build submission: Healthcare Offer Prioritization

**Industry:** Healthcare payer (health plan), member engagement and care management.
**Author:** Bhavik Bhatt. **Data:** 100% synthetic; no customer or real member data is used anywhere.

## The customer problem

A health plan runs a catalog of care and wellness programs (diabetes and heart-health management,
mental health support, cancer screening, generic-drug switching, HSA/FSA optimization and so on). Today
the plan picks which program to offer each member with static rules ("age > 65 and diabetic → program X")
and generic outreach. The result:

* **Low program enrollment.** Members get offers that don't match where they are in their health journey,
  and ignore them. Every member who should be in a chronic-disease program but isn't is avoidable cost of care.
* **Care-manager time lost to prep.** Before a call, a care manager hunts across claims, eligibility and
  CRM screens to decide what to offer and why.
* **No "why."** Even when a recommendation is right, nobody can explain it to the member, so it isn't trusted.

**What the build does:** every member gets all 16 programs ranked by an ML model, the top 5 are served to
the care manager's console with a plain-language reason written by an LLM from the member's profile and the
model's SHAP drivers, the care manager approves or rejects each one (feedback is captured for retraining),
and business users can ask questions of the same governed data in natural language through a Genie Agent.

**Buyer KPIs** (quantified in the business deck, text version at [`deck/FE_Bar_Deck.md`](deck/FE_Bar_Deck.md)): care-program enrollment, care-manager minutes per case, voluntary
churn, outreach cost per engaged member, and Stars/HEDIS gap closure for preventive programs.

## One integrated journey: where each stage is, and the proof it ran

Everything is deployed by one Databricks Asset Bundle ([`databricks.yml`](databricks.yml)) and orchestrated by
one serverless Lakeflow Job, `offer-prioritization-e2e`, whose tasks hand off through Unity Catalog tables.
**Execution evidence is committed as text** in [`evidence/`](evidence/). It was exported from job run
`923080212888235` on 2026-10-27 (serverless, ~11 min end to end) with [`scripts/export_run_evidence.py`](scripts/export_run_evidence.py) and
[`scripts/collect_platform_evidence.py`](scripts/collect_platform_evidence.py): the executed notebooks, with
their real cell outputs, are in both `.ipynb` and readable `.md` form.

| Stage | What it does | Code | Execution evidence (text) |
| --- | --- | --- | --- |
| **0. Raw data** | Generates synthetic eligibility, claims, benefits and CRM-engagement extracts and lands them as JSON in a UC Volume, with ~0.5% realistic defects injected | [`src/jobs/00_land_raw_data.py`](src/jobs/00_land_raw_data.py), [`data/generate_synthetic_data.py`](data/generate_synthetic_data.py) | [`evidence/notebooks/land_raw_data.md`](evidence/notebooks/land_raw_data.md): 5,000 members, 21,350 claims, 35,000 benefit rows and 63,175 engagement events landed, with 482 defective records injected on purpose |
| **1. Lakeflow** | Spark Declarative Pipeline: Auto Loader bronze → typed silver with 15 data-quality expectations → 3 gold materialized views (member 360, monthly claims, outreach performance) | [`src/pipeline/transformations/`](src/pipeline/transformations/), [`resources/offer_medallion.pipeline.yml`](resources/offer_medallion.pipeline.yml), [`resources/offer_e2e.job.yml`](resources/offer_e2e.job.yml) | [`evidence/pipeline_update.md`](evidence/pipeline_update.md): rows per flow and expectation pass/fail from the pipeline event log. The silver expectations dropped exactly the 408 injected bad rows (20 members, 128 claims, 70 benefit rows, 190 engagement events) and flagged 159 warn-only violations; gold_member_360 = 4,980 members |
| **2. Unity Catalog** | All assets in `bhavik_demos.offer_prioritization`: table and column comments, governed tags (`domain=Healthcare`, `phi`; `member_id` tagged as the HIPAA `health_plan_beneficiary_number`), least-privilege grants (the app SP can only write `offer_feedback`), model registry, lineage | [`src/jobs/05_uc_governance.py`](src/jobs/05_uc_governance.py) | [`evidence/notebooks/uc_governance.md`](evidence/notebooks/uc_governance.md) (grants and tags applied), [`evidence/platform.md`](evidence/platform.md) (inventory, tags, lineage from `system.access.table_lineage`, model versions) |
| **3. ML + Gen AI** | 96 features from silver → multi-output LightGBM (one score per offer) tracked in MLflow and registered in UC as `@champion` → batch scoring of every member → SHAP drivers → `databricks-gpt-5-1` writes the "why", checked by guardrails | [`src/jobs/02_train_register_model.py`](src/jobs/02_train_register_model.py), [`src/jobs/03_batch_inference_reasoning.py`](src/jobs/03_batch_inference_reasoning.py), [`features/`](features/feature_engineering.py), [`models/`](models/offer_model.py) | [`evidence/notebooks/train_register_model.md`](evidence/notebooks/train_register_model.md): 96 features, model v3 `@champion`, test R² 0.784, RMSE 5.10, top-5 overlap 0.846, top-1 accuracy 0.815. [`evidence/notebooks/batch_inference_reasoning.md`](evidence/notebooks/batch_inference_reasoning.md): 4,980 members x 16 offers scored (24,900 top-5 rows); 500 explanations, 100% LLM-generated, 0% mention scores or models, 93.8% name the program, median 89 words |
| **4. Lakebase** | Publishes the care-manager work queue into Lakebase Postgres (atomic staging-table swap, primary key, index, SP grant) for the app's low-latency reads | [`src/jobs/04_serve_lakebase.py`](src/jobs/04_serve_lakebase.py) | [`evidence/notebooks/serve_lakebase.md`](evidence/notebooks/serve_lakebase.md): 500 rows / 100 members published; member lookup p50 3.1 ms, p95 7.2 ms over 200 queries; EXPLAIN shows a bitmap index scan on `idx_lakebase_offers_member` |
| **5. Genie Agent** | "Member Offer Intelligence": 7 governed tables, business instructions, join specs, example SQL, benchmarks; created via API, then asked 5 business questions | [`src/jobs/06_genie_agent.py`](src/jobs/06_genie_agent.py) | [`evidence/notebooks/genie_agent.md`](evidence/notebooks/genie_agent.md): every question with the SQL Genie generated, the result rows and its answer (5/5 answered with executed SQL and result rows) |
| **6. Databricks App** | "Offer Management Console" (Dash): member lookup from Lakebase, top 5 offers with AI reasoning and SHAP "decision factors", approve/reject plus comments written to the governed Delta table through parameterized SQL | [`app/app.py`](app/app.py), [`app/app.yaml`](app/app.yaml) | [`evidence/app_e2e_test.md`](evidence/app_e2e_test.md): live UI test, member lookup and approve, then the resulting `offer_feedback` row queried back from UC; app status in [`evidence/platform.md`](evidence/platform.md) |

Job-level summary (every task, state, duration, notebook exit values): [`evidence/job_run.json`](evidence/job_run.json).

## How it fits together

```
 UC Volume raw_landing (JSON extracts, synthetic)
        │  Auto Loader
        ▼
 Lakeflow pipeline ── bronze_* ── silver_* (expectations) ── gold_member_360 / gold_claims_monthly / gold_engagement_performance
        │                                │                                   │
        │                                ▼                                   │
        │                 member_features (96 features)                      │
        │                                ▼                                   │
        │        LightGBM x16 ── MLflow ── UC model @champion                │
        │                                ▼                                   │
        │   offer_recommendations (all members) ── SHAP ── GPT-5.1 reasoning │
        │                                ▼                                   ▼
        │        member_offer_recommendations_with_reasoning ──────► Genie Agent "Member Offer Intelligence"
        │                                │                                   ▲
        │                                ▼ (job publishes)                   │
        │                Lakebase Postgres lakebase_offers                   │
        │                                ▼ (p95 lookups in milliseconds)     │
        │                Databricks App "Offer Management Console" ──► offer_feedback (Delta, UC) ──► retraining
        └──────────── all governed in Unity Catalog: comments, governed tags, grants, lineage, model registry ─────┘
```

## Design decisions and trade-offs (AI mindset)

* **Rule labels, stated honestly.** There is no historical offer-acceptance label in a synthetic dataset,
  so the training target is a clinically motivated targeting score per offer (`RuleBasedScorer`). The model
  learns a smooth ranking over all 96 features instead of brittle if/then rules, but its test metrics
  measure fit to those rules, not real-world acceptance. The `offer_feedback` table the app writes is the
  path to replacing rule labels with observed outcomes; `notebooks/05_model_retraining.py` shows
  champion/challenger retraining. The deck's value model is therefore framed as a pilot hypothesis with a
  control group and decision gate, not a claim.
* **Ranking metrics over regression metrics.** The care manager sees a top-5 list, so beyond R² I report
  top-5 overlap and top-1 accuracy against the target ranking.
* **LLM only where it adds value.** The model decides; the LLM explains. It is grounded in the member profile
  and the SHAP drivers for that specific score, prompted never to cite scores, models or give medical advice,
  and every output is checked (share LLM-generated vs fallback, jargon leakage, names the program, length).
  Reasoning is generated for the care-manager work queue (100 members, 500 offers), not all members, to
  bound cost and latency.
* **Lakebase for the hot path, Delta for the truth.** The console needs single-member lookups in
  milliseconds; it reads Lakebase. Feedback is analytical ground truth for retraining, so it is written to
  a governed Delta table. The Lakebase load is an atomic staging-table swap, so the app never sees a
  half-loaded table.
* **Data quality is visible.** About 0.5% of raw records are deliberately corrupted, the way real extracts
  are. Silver expectations drop or flag them, and the counts are in the pipeline event log.
* **Governance follows the workspace's tag policies.** `domain` and `phi` are governed tags in this
  metastore, so their values come from the allowed lists (`Healthcare`; `health_plan_beneficiary_number`).
* **Security.** The app's feedback writes use Statement Execution API named parameters (no string-spliced
  SQL). Lakebase and SQL calls use short-lived OAuth tokens; no secrets are stored.

## Known limitations and next steps

* Each job run is one synthetic extract batch (new member IDs). Gold views and the model use the latest
  `batch_id`. Real extracts would carry stable member IDs and the silver layer would use `AUTO CDC` (SCD2).
* The Lakebase load is a scheduled publish from Delta. With a dedicated catalog, a Lakebase synced table
  (continuous) would replace it.
* Column masks and row filters on PHI columns (keyed off the `phi` tags) are the next governance step
  before real data.
* Lineage from silver into `member_features` goes through pandas, so UC records it as job read/write events
  rather than a column-level edge.

## Reproduce

```bash
databricks bundle deploy -t dev --profile <profile>
databricks bundle run offer_prioritization_e2e -t dev --profile <profile>
python -I scripts/export_run_evidence.py --profile <profile> --run-id <job_run_id>
python -I scripts/collect_platform_evidence.py --profile <profile> --since <YYYY-MM-DD> --genie-space-id <id>
```

The original exploratory notebooks (`notebooks/01`–`05`) are kept for reference. The production path
used for this submission is `src/` plus `resources/`.
