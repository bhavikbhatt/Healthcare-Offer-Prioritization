# Executed notebook: `train_register_model`

- Job run task id: `934034849659822`
- Result: **SUCCESS**
- Started: 2026-10-27T20:20:59.123000+00:00
- Duration: 147 s

# 02 · Features + model training + Unity Catalog registration

Reads the **Lakeflow silver tables** (latest extract batch), engineers 80+ member
features, persists them as the governed `member_features` table, trains a
multi-output LightGBM model (one score per offer, 16 offers), tracks it in MLflow
and registers it in Unity Catalog with the `@champion` alias.

**Honest caveat:** there is no historical offer-acceptance label in a synthetic
dataset, so the training target is a clinically-motivated rule score per offer
(`RuleBasedScorer`). The model generalises those rules into a smooth ranking over
all features; approve/reject feedback captured in the app (`offer_feedback`) is the
path to replacing rule labels with observed outcomes (see `notebooks/05_model_retraining.py`).

```python
%pip install lightgbm mlflow --quiet
```
Output:
```text
Note: you may need to restart the kernel using %restart_python or dbutils.library.restartPython() to use updated packages.
```

```python
dbutils.library.restartPython()
```

```python
dbutils.widgets.text("catalog", "bhavik_demos")
dbutils.widgets.text("schema", "offer_prioritization")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FQ = f"{CATALOG}.{SCHEMA}"
UC_MODEL = f"{FQ}.healthcare_offer_prioritizer"
EXPERIMENT = "/Shared/healthcare_offer_prioritization"
```

```python
%run ../../features/feature_engineering
```

```python

```

```python

```

```python

```

```python
%run ../../models/offer_model
```

```python

```

```python

```

```python

```

```python

```

```python

```

```python
import json
import numpy as np
import pandas as pd

batch_id = spark.sql(f"SELECT max(batch_id) FROM {FQ}.silver_members").first()[0]
print(f"Training on extract batch {batch_id}")

def latest(table, cols):
    return spark.sql(f"SELECT {', '.join(cols)} FROM {FQ}.{table} WHERE batch_id = '{batch_id}'").toPandas()

members_df = latest("silver_members", [
    "member_id", "age", "gender", "region", "plan_type", "tenure_months", "income_bracket",
    "family_size", "enrollment_date", "risk_score", "chronic_condition_count"])
claims_df = latest("silver_claims", [
    "claim_id", "member_id", "claim_date", "claim_type", "diagnosis_category", "claim_amount",
    "allowed_amount", "paid_amount", "member_responsibility", "provider_type", "in_network", "status"])
benefits_df = latest("silver_benefits_utilization", [
    "member_id", "benefit_type", "annual_max", "used_amount", "remaining_balance",
    "utilization_rate", "last_used_date", "claims_count", "plan_year"])
engagement_df = latest("silver_engagement_history", [
    "engagement_id", "member_id", "engagement_date", "channel", "engagement_type",
    "response_flag", "session_duration_sec", "offer_id"])

for name, pdf in [("members", members_df), ("claims", claims_df),
                  ("benefits", benefits_df), ("engagement", engagement_df)]:
    print(f"  silver {name:<11} {len(pdf):>8,} rows")
```
Output:
```text
Training on extract batch 20261027T201919Z
  silver members        4,980 rows
  silver claims        21,222 rows
  silver benefits      34,930 rows
  silver engagement    62,985 rows
```

 ## Feature engineering → `member_features` (gold, governed in UC)

```python
features_df, feature_engineer = create_member_features(members_df, claims_df, benefits_df, engagement_df)
feature_names = [c for c in features_df.columns if c != "member_id"]
print(f"{len(features_df):,} members x {len(feature_names)} features")

(spark.createDataFrame(features_df.assign(batch_id=batch_id))
    .write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{FQ}.member_features"))
print(f"wrote {FQ}.member_features")

# Offer catalog as a governed reference table (joined by Genie and the app)
catalog = OfferCatalog()
offer_catalog_pdf = pd.DataFrame([{
    "offer_id": o["offer_id"], "offer_name": o["name"], "category": o["category"],
    "base_score": float(o["base_score"]), "targeting_rules": json.dumps(o["targeting_rules"]),
} for o in catalog.offers])
spark.createDataFrame(offer_catalog_pdf).write.mode("overwrite").option("overwriteSchema", "true") \
    .saveAsTable(f"{FQ}.offer_catalog")
print(f"wrote {FQ}.offer_catalog ({len(offer_catalog_pdf)} offers)")
display(offer_catalog_pdf[["offer_id", "offer_name", "category", "base_score"]])
```
Output:
```text
🔧 Starting Feature Engineering Pipeline
============================================================

📊 Creating demographic features...
   ✓ 15 features

🏥 Creating claims features...
   ✓ 26 features

🩺 Creating diagnosis features...
   ✓ 10 features

💊 Creating benefits features...
   ✓ 16 features

📱 Creating engagement features...
   ✓ 21 features

📅 Creating temporal features...
   ✓ 8 features

🔗 Merging feature groups...

✅ Feature engineering complete!
   Total features: 96
   Total members: 4,980
============================================================
4,980 members x 96 features
wrote bhavik_demos.offer_prioritization.member_features
wrote bhavik_demos.offer_prioritization.offer_catalog (16 offers)

offer_id | offer_name | category | base_score
--- | --- | --- | ---
PREV_001 | Annual Wellness Visit Reminder | preventive_care | 60.0
PREV_002 | Cancer Screening Program | preventive_care | 55.0
CHRON_001 | Diabetes Management Program | chronic_disease | 50.0
CHRON_002 | Heart Health Program | chronic_disease | 50.0
CHRON_003 | Respiratory Care Program | chronic_disease | 50.0
MH_001 | Mental Health Support | mental_health | 55.0
MH_002 | Stress Management Workshop | mental_health | 50.0
PHARM_001 | Generic Drug Switch Program | pharmacy | 45.0
PHARM_002 | Mail-Order Pharmacy | pharmacy | 45.0
TELE_001 | Virtual Primary Care | telehealth | 50.0
TELE_002 | Virtual Specialist Consultations | telehealth | 45.0
FIT_001 | Gym Membership Discount | fitness | 40.0
FIT_002 | Nutrition Coaching | fitness | 45.0
NAV_001 | Care Navigator Assignment | care_navigation | 40.0
COST_001 | HSA/FSA Optimization | cost_savings | 45.0
COST_002 | In-Network Provider Finder | cost_savings | 45.0
```

 ## Train + track in MLflow + register in Unity Catalog

```python
import mlflow
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXPERIMENT)

X, y = create_training_data(features_df, catalog)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

MODEL_PARAMS = {
    "objective": "regression", "metric": "rmse", "boosting_type": "gbdt",
    "num_leaves": 31, "learning_rate": 0.05, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 5, "verbose": -1,
    "n_estimators": 200, "random_state": 42,
}

def top_k_overlap(y_true, y_pred, k=5):
    """Share of each member's true top-k offers that the model also ranks top-k."""
    t = np.argsort(-y_true, axis=1)[:, :k]
    p = np.argsort(-y_pred, axis=1)[:, :k]
    return float(np.mean([len(set(a) & set(b)) / k for a, b in zip(t, p)]))

with mlflow.start_run(run_name=f"offer_prioritizer_{batch_id}") as run:
    model = OfferPrioritizationModel(catalog=catalog, model_params=MODEL_PARAMS)
    train_metrics = model.fit(X_train, y_train, feature_names=feature_names)

    # Ship scaler + multi-output LightGBM as ONE sklearn Pipeline so the
    # UC-registered model is self-contained for batch inference.
    pipeline = Pipeline([("scaler", model.scaler), ("model", model.model)])
    y_pred = np.clip(pipeline.predict(X_test), 0, 100)

    metrics = {
        "train_rmse": train_metrics["rmse"],
        "test_rmse": float(np.sqrt(mean_squared_error(y_test, y_pred))),
        "test_mae": float(mean_absolute_error(y_test, y_pred)),
        "test_r2": float(r2_score(y_test, y_pred)),
        "test_top5_overlap": top_k_overlap(y_test.values, y_pred, 5),
        "test_top1_accuracy": float(np.mean(np.argmax(y_test.values, 1) == np.argmax(y_pred, 1))),
    }
    per_offer_rmse = {
        f"rmse_{oid}": float(np.sqrt(mean_squared_error(y_test.iloc[:, i], y_pred[:, i])))
        for i, oid in enumerate(catalog.get_offer_ids())
    }
    mlflow.log_params({
        "model_type": "LightGBM MultiOutputRegressor", "n_features": len(feature_names),
        "n_offers": y.shape[1], "n_train": len(X_train), "n_test": len(X_test),
        "source_batch_id": batch_id, "source_tables": f"{FQ}.silver_*",
        **{k: v for k, v in MODEL_PARAMS.items() if k in ("n_estimators", "learning_rate", "num_leaves")},
    })
    mlflow.log_metrics({**metrics, **per_offer_rmse})
    mlflow.log_dict({"offer_ids": catalog.get_offer_ids(), "feature_names": feature_names},
                    "model_metadata.json")
    mlflow.sklearn.log_model(
        pipeline, artifact_path="model",
        signature=infer_signature(X_train, y_pred),
        input_example=X_train.head(3),
        serialization_format="cloudpickle",
    )
    run_id = run.info.run_id

print(json.dumps(metrics, indent=2))
```
Output:
```text
If you are using MLflow Tracing, you can migrate your traces to Unity Catalog for unlimited storage, fine-grained access controls, and queryability from notebooks, SQL, and dashboards. Learn more: https://docs.databricks.com/aws/en/mlflow3/genai/tracing/migrate-traces-to-uc
2026/10/27 20:22:48 WARNING mlflow.models.model: `artifact_path` is deprecated. Please use `name` instead.
🔗 View Logged Model at: https://fevm-classic-stable-tp909z.cloud.databricks.com/ml/experiments/1743891818208117/models/m-783fe5e4d65d497e89e6eaa42d1efcc1?o=7474656617733963

{
  "train_rmse": 2.9010447051004307,
  "test_rmse": 5.102058931073664,
  "test_mae": 4.081618628359976,
  "test_r2": 0.7836007792770753,
  "test_top5_overlap": 0.8457831325301206,
  "test_top1_accuracy": 0.8152610441767069
}
```

```python
mv = mlflow.register_model(f"runs:/{run_id}/model", UC_MODEL)
client = MlflowClient()
client.set_registered_model_alias(UC_MODEL, "champion", mv.version)
client.update_model_version(
    UC_MODEL, mv.version,
    description=f"Trained on extract batch {batch_id}; test R2={metrics['test_r2']:.3f}, "
                f"top-5 overlap={metrics['test_top5_overlap']:.3f}")
print(f"Registered {UC_MODEL} v{mv.version} -> @champion  (run {run_id})")
```
Output:
```text
Registered model 'bhavik_demos.offer_prioritization.healthcare_offer_prioritizer' already exists. Creating a new version of this model...
2026/10/27 20:23:13 WARNING mlflow.tracking._model_registry.fluent: Run with id 4c2ee4573c4f4a7e990c704ddb057f4e has no artifacts at artifact path 'model', registering model based on models:/m-783fe5e4d65d497e89e6eaa42d1efcc1 instead

🔗 Created version '3' of model 'bhavik_demos.offer_prioritization.healthcare_offer_prioritizer': https://fevm-classic-stable-tp909z.cloud.databricks.com/explore/data/models/bhavik_demos/offer_prioritization/healthcare_offer_prioritizer/version/3?o=7474656617733963

Registered bhavik_demos.offer_prioritization.healthcare_offer_prioritizer v3 -> @champion  (run 4c2ee4573c4f4a7e990c704ddb057f4e)
```

```python
importance = pd.DataFrame({
    "feature": feature_names,
    "importance": np.mean([est.feature_importances_ for est in model.model.estimators_], axis=0),
}).sort_values("importance", ascending=False)
spark.createDataFrame(importance.assign(model_version=str(mv.version))) \
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{FQ}.model_feature_importance")
print("Top 15 features by mean gain across the 16 offer models:")
print(importance.head(15).to_string(index=False))
```
Output:
```text
Top 15 features by mean gain across the 16 offer models:
                       feature  importance
                    risk_score    199.3750
       vision_utilization_rate    189.0000
       dental_utilization_rate    181.4375
   telehealth_utilization_rate    180.6250
     pharmacy_utilization_rate    177.9375
mental_health_utilization_rate    170.3750
                           age    164.2500
   preventive_utilization_rate    162.4375
          std_utilization_rate    159.8125
                 tenure_months    159.6875
          avg_session_duration    158.2500
    days_since_last_engagement    154.5000
         days_since_last_claim    152.3125
          avg_utilization_rate    148.3125
             avg_response_rate    145.5625
```

```python
dbutils.notebook.exit(json.dumps({
    "batch_id": batch_id, "run_id": run_id, "model": UC_MODEL,
    "version": str(mv.version), "metrics": metrics,
}))
```
Output:
```text
{"batch_id": "20261027T201919Z", "run_id": "4c2ee4573c4f4a7e990c704ddb057f4e", "model": "bhavik_demos.offer_prioritization.healthcare_offer_prioritizer", "version": "3", "metrics": {"train_rmse": 2.9010447051004307, "test_rmse": 5.102058931073664, "test_mae": 4.081618628359976, "test_r2": 0.7836007792770753, "test_top5_overlap": 0.8457831325301206, "test_top1_accuracy": 0.8152610441767069}}
```
