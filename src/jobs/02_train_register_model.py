# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Features + model training + Unity Catalog registration
# MAGIC
# MAGIC Reads the **Lakeflow silver tables** (latest extract batch), engineers 80+ member
# MAGIC features, persists them as the governed `member_features` table, trains a
# MAGIC multi-output LightGBM model (one score per offer, 16 offers), tracks it in MLflow
# MAGIC and registers it in Unity Catalog with the `@champion` alias.
# MAGIC
# MAGIC **Honest caveat:** there is no historical offer-acceptance label in a synthetic
# MAGIC dataset, so the training target is a clinically-motivated rule score per offer
# MAGIC (`RuleBasedScorer`). The model generalises those rules into a smooth ranking over
# MAGIC all features; approve/reject feedback captured in the app (`offer_feedback`) is the
# MAGIC path to replacing rule labels with observed outcomes (see `notebooks/05_model_retraining.py`).

# COMMAND ----------

# MAGIC %pip install lightgbm mlflow --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

dbutils.widgets.text("catalog", "bhavik_demos")
dbutils.widgets.text("schema", "offer_prioritization")

CATALOG = dbutils.widgets.get("catalog")
SCHEMA = dbutils.widgets.get("schema")
FQ = f"{CATALOG}.{SCHEMA}"
UC_MODEL = f"{FQ}.healthcare_offer_prioritizer"
EXPERIMENT = "/Shared/healthcare_offer_prioritization"

# COMMAND ----------

# MAGIC %run ../../features/feature_engineering

# COMMAND ----------

# MAGIC %run ../../models/offer_model

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md ## Feature engineering → `member_features` (gold, governed in UC)

# COMMAND ----------

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

# COMMAND ----------

# MAGIC %md ## Train + track in MLflow + register in Unity Catalog

# COMMAND ----------

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

# COMMAND ----------

mv = mlflow.register_model(f"runs:/{run_id}/model", UC_MODEL)
client = MlflowClient()
client.set_registered_model_alias(UC_MODEL, "champion", mv.version)
client.update_model_version(
    UC_MODEL, mv.version,
    description=f"Trained on extract batch {batch_id}; test R2={metrics['test_r2']:.3f}, "
                f"top-5 overlap={metrics['test_top5_overlap']:.3f}")
print(f"Registered {UC_MODEL} v{mv.version} -> @champion  (run {run_id})")

# COMMAND ----------

importance = pd.DataFrame({
    "feature": feature_names,
    "importance": np.mean([est.feature_importances_ for est in model.model.estimators_], axis=0),
}).sort_values("importance", ascending=False)
spark.createDataFrame(importance.assign(model_version=str(mv.version))) \
    .write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(f"{FQ}.model_feature_importance")
print("Top 15 features by mean gain across the 16 offer models:")
print(importance.head(15).to_string(index=False))

# COMMAND ----------

dbutils.notebook.exit(json.dumps({
    "batch_id": batch_id, "run_id": run_id, "model": UC_MODEL,
    "version": str(mv.version), "metrics": metrics,
}))
