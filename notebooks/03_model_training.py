# Databricks notebook source
# MAGIC %md
# MAGIC # 03 - Model Training with MLflow
# MAGIC 
# MAGIC This notebook trains the offer prioritization model and registers it in Databricks MLflow.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Tuple

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from mlflow.models.signature import infer_signature

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# MLflow experiment configuration
EXPERIMENT_NAME = "/Shared/healthcare_offer_prioritization"
MODEL_NAME = "healthcare_offer_prioritizer"
CATALOG_NAME = "healthcare_demo"  # For Unity Catalog model registry
SCHEMA_NAME = "offer_prioritization"

# Model hyperparameters
MODEL_PARAMS = {
    "objective": "regression",
    "metric": "rmse",
    "boosting_type": "gbdt",
    "num_leaves": 31,
    "learning_rate": 0.05,
    "feature_fraction": 0.8,
    "bagging_fraction": 0.8,
    "bagging_freq": 5,
    "verbose": -1,
    "n_estimators": 200,
    "random_state": 42
}

# Training settings
TEST_SIZE = 0.2
RANDOM_STATE = 42

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data and Create Features

# COMMAND ----------

# Generate synthetic data for demo
try:
    from data.generate_synthetic_data import generate_all_data
except ImportError:
    exec(open("../data/generate_synthetic_data.py").read())

try:
    from features.feature_engineering import create_member_features
except ImportError:
    exec(open("../features/feature_engineering.py").read())

try:
    from models.offer_model import (
        OfferPrioritizationModel, 
        OfferCatalog, 
        RuleBasedScorer,
        create_training_data
    )
except ImportError:
    exec(open("../models/offer_model.py").read())

# COMMAND ----------

# Generate data
print("🔄 Generating synthetic data...")
members_df, claims_df, benefits_df, engagement_df = generate_all_data(
    n_members=50000,
    seed=42
)

# Create features
print("\n🔧 Creating features...")
features_df, feature_engineer = create_member_features(
    members_df=members_df,
    claims_df=claims_df,
    benefits_df=benefits_df,
    engagement_df=engagement_df
)

print(f"\n✅ Features created: {len(feature_engineer.feature_names)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Create Training Data

# COMMAND ----------

# Initialize offer catalog and scorer
catalog = OfferCatalog()
print(f"📋 Offer Catalog: {len(catalog.offers)} offers")

# Create training data with target scores
X, y = create_training_data(features_df, catalog)

print(f"\n📊 Training Data:")
print(f"   X shape: {X.shape}")
print(f"   y shape: {y.shape}")
print(f"   Target columns: {list(y.columns)[:5]}... ({len(y.columns)} total)")

# COMMAND ----------

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE
)

print(f"\n📊 Data Split:")
print(f"   Training: {len(X_train):,} samples")
print(f"   Testing: {len(X_test):,} samples")

# COMMAND ----------

# MAGIC %md
# MAGIC ## MLflow Experiment Setup

# COMMAND ----------

# Set MLflow experiment
# In Databricks, this will create the experiment in the workspace
mlflow.set_experiment(EXPERIMENT_NAME)

print(f"✅ MLflow Experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Training with MLflow Tracking

# COMMAND ----------

def train_and_log_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    model_params: Dict,
    feature_names: List[str],
    run_name: str = "offer_prioritization_model"
) -> Tuple[OfferPrioritizationModel, str]:
    """
    Train model and log to MLflow
    
    Returns:
        Tuple of (trained_model, run_id)
    """
    
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"\n🚀 MLflow Run Started: {run_id}")
        
        # Log parameters
        mlflow.log_params({
            "model_type": "LightGBM MultiOutput",
            "n_estimators": model_params.get("n_estimators", 200),
            "learning_rate": model_params.get("learning_rate", 0.05),
            "num_leaves": model_params.get("num_leaves", 31),
            "feature_fraction": model_params.get("feature_fraction", 0.8),
            "test_size": TEST_SIZE,
            "n_features": len(feature_names),
            "n_offers": y_train.shape[1],
            "n_training_samples": len(X_train)
        })
        
        # Log dataset info
        mlflow.log_param("feature_names", str(feature_names[:20]) + "...")
        mlflow.log_param("target_columns", str(list(y_train.columns)[:5]) + "...")
        
        # Train model
        print("\n📈 Training model...")
        model = OfferPrioritizationModel(catalog=catalog, model_params=model_params)
        train_metrics = model.fit(X_train, y_train, feature_names=feature_names)
        
        # Log training metrics
        for metric_name, metric_value in train_metrics.items():
            mlflow.log_metric(f"train_{metric_name}", metric_value)
        print(f"   Training RMSE: {train_metrics['rmse']:.4f}")
        print(f"   Training R²: {train_metrics['r2']:.4f}")
        
        # Evaluate on test set
        print("\n📊 Evaluating on test set...")
        raw_scores, _ = model.predict(X_test)
        
        test_metrics = {
            "rmse": np.sqrt(mean_squared_error(y_test, raw_scores)),
            "mae": mean_absolute_error(y_test, raw_scores),
            "r2": r2_score(y_test, raw_scores)
        }
        
        # Log test metrics
        for metric_name, metric_value in test_metrics.items():
            mlflow.log_metric(f"test_{metric_name}", metric_value)
        print(f"   Test RMSE: {test_metrics['rmse']:.4f}")
        print(f"   Test R²: {test_metrics['r2']:.4f}")
        
        # Get and log feature importance
        importance_df = model.get_feature_importance()
        importance_df.to_csv("/tmp/feature_importance.csv", index=False)
        mlflow.log_artifact("/tmp/feature_importance.csv")
        
        # Create feature importance plot
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(20)
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importances')
        plt.tight_layout()
        plt.savefig("/tmp/feature_importance.png", dpi=150, bbox_inches='tight')
        mlflow.log_artifact("/tmp/feature_importance.png")
        plt.close()
        
        # Create prediction distribution plot
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        axes = axes.flatten()
        
        for i, (ax, col) in enumerate(zip(axes, y_test.columns[:6])):
            ax.hist(y_test[col], bins=30, alpha=0.5, label='Actual', color='blue')
            ax.hist(raw_scores[:, i], bins=30, alpha=0.5, label='Predicted', color='orange')
            ax.set_title(col.replace('score_', ''))
            ax.legend()
        
        plt.tight_layout()
        plt.savefig("/tmp/prediction_distribution.png", dpi=150, bbox_inches='tight')
        mlflow.log_artifact("/tmp/prediction_distribution.png")
        plt.close()
        
        # Create model signature
        signature = infer_signature(X_train, raw_scores)
        
        # Log the model
        print("\n📦 Logging model to MLflow...")
        
        # Log as sklearn model (works with MultiOutputRegressor)
        mlflow.sklearn.log_model(
            model.model,
            artifact_path="model",
            signature=signature,
            registered_model_name=None  # Will register separately
        )
        
        # Also save the full model wrapper for inference
        import joblib
        model.save("/tmp/offer_model.joblib")
        mlflow.log_artifact("/tmp/offer_model.joblib")
        
        # Log additional metadata
        mlflow.log_dict({
            "offer_ids": catalog.get_offer_ids(),
            "feature_names": feature_names,
            "model_params": model_params
        }, "model_metadata.json")
        
        print(f"\n✅ MLflow Run Complete: {run_id}")
        
        return model, run_id

# COMMAND ----------

# Train and log model
feature_names = list(X.columns)
model, run_id = train_and_log_model(
    X_train, X_test, y_train, y_test,
    MODEL_PARAMS, feature_names,
    run_name="offer_prioritization_v1"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Model in Databricks Model Registry

# COMMAND ----------

def register_model_to_registry(
    run_id: str,
    model_name: str,
    catalog_name: str = None,
    schema_name: str = None,
    description: str = None
) -> str:
    """
    Register model to Databricks Model Registry
    
    For Unity Catalog, use: catalog_name.schema_name.model_name
    For Workspace registry, use just: model_name
    """
    
    # Construct model path
    if catalog_name and schema_name:
        # Unity Catalog model registry
        full_model_name = f"{catalog_name}.{schema_name}.{model_name}"
    else:
        # Workspace model registry
        full_model_name = model_name
    
    print(f"\n📝 Registering model: {full_model_name}")
    
    # Get the run's model artifact URI
    model_uri = f"runs:/{run_id}/model"
    
    # Register the model
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=full_model_name
    )
    
    print(f"✅ Model registered: {full_model_name}")
    print(f"   Version: {model_version.version}")
    
    # Update model description if provided
    if description:
        from mlflow.tracking import MlflowClient
        client = MlflowClient()
        client.update_registered_model(
            name=full_model_name,
            description=description
        )
    
    return model_version.version

# COMMAND ----------

# Register model
# For Unity Catalog (recommended in Databricks):
# model_version = register_model_to_registry(
#     run_id=run_id,
#     model_name=MODEL_NAME,
#     catalog_name=CATALOG_NAME,
#     schema_name=SCHEMA_NAME,
#     description="Healthcare offer prioritization model using LightGBM"
# )

# For Workspace registry (simpler setup):
model_version = register_model_to_registry(
    run_id=run_id,
    model_name=MODEL_NAME,
    description="Healthcare offer prioritization model that predicts priority scores for 16 healthcare offers based on member demographics, claims, benefits, and engagement data."
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Validation

# COMMAND ----------

# Sample predictions
print("\n🔍 Sample Predictions:")
print("=" * 60)

# Get predictions for first 5 test samples
sample_X = X_test.head(5)
sample_scores, top_offers = model.predict(sample_X, return_top_n=3)

# Display results
for i in range(5):
    member_offers = top_offers[top_offers['member_idx'] == i]
    print(f"\nMember {i+1}:")
    print(f"  Age: {sample_X.iloc[i].get('age', 'N/A')}")
    print(f"  Top 3 Offers:")
    for _, row in member_offers.iterrows():
        print(f"    {row['rank']}. {row['offer_name']} (Score: {row['priority_score']:.1f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Importance Analysis

# COMMAND ----------

# Display top features
importance_df = model.get_feature_importance()

print("\n📊 Top 20 Most Important Features:")
print("=" * 50)
for i, row in importance_df.head(20).iterrows():
    bar = "█" * int(row['importance'] / importance_df['importance'].max() * 30)
    print(f"{row['feature'][:30]:30s} {bar} {row['importance']:.4f}")

# COMMAND ----------

# Visualize feature importance
plt.figure(figsize=(12, 10))
top_n = 25
top_features = importance_df.head(top_n)

colors = plt.cm.viridis(np.linspace(0.3, 0.9, top_n))
plt.barh(range(top_n), top_features['importance'].values, color=colors)
plt.yticks(range(top_n), top_features['feature'].values)
plt.xlabel('Feature Importance', fontsize=12)
plt.title('Top 25 Features for Offer Prioritization', fontsize=14, fontweight='bold')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Model Performance by Offer Category

# COMMAND ----------

# Get predictions for full test set
raw_scores, _ = model.predict(X_test)

# Calculate RMSE by offer
offer_rmse = {}
for i, col in enumerate(y_test.columns):
    rmse = np.sqrt(mean_squared_error(y_test[col], raw_scores[:, i]))
    offer_id = col.replace('score_', '')
    offer = catalog.get_offer_by_id(offer_id)
    offer_rmse[offer['name'] if offer else offer_id] = rmse

# Sort by RMSE
offer_rmse_sorted = dict(sorted(offer_rmse.items(), key=lambda x: x[1]))

# Visualize
plt.figure(figsize=(12, 8))
names = list(offer_rmse_sorted.keys())
rmses = list(offer_rmse_sorted.values())

plt.barh(names, rmses, color='steelblue')
plt.xlabel('RMSE', fontsize=12)
plt.title('Model RMSE by Offer', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("\n" + "=" * 60)
print("📊 MODEL TRAINING SUMMARY")
print("=" * 60)
print(f"\n🎯 Experiment: {EXPERIMENT_NAME}")
print(f"🏷️  Run ID: {run_id}")
print(f"📦 Model: {MODEL_NAME} (version {model_version})")
print(f"\n📈 Performance Metrics:")
print(f"   Test RMSE: {np.sqrt(mean_squared_error(y_test, raw_scores)):.4f}")
print(f"   Test MAE: {mean_absolute_error(y_test, raw_scores):.4f}")
print(f"   Test R²: {r2_score(y_test, raw_scores):.4f}")
print(f"\n🔢 Model Details:")
print(f"   Features: {len(feature_names)}")
print(f"   Offers: {len(catalog.offers)}")
print(f"   Training samples: {len(X_train):,}")
print(f"\n✅ Model registered and ready for inference (Notebook 04)")
print("=" * 60)

