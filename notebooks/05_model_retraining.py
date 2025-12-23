# Databricks notebook source
# MAGIC %md
# MAGIC # 05 - Model Retraining with Feedback
# MAGIC
# MAGIC This notebook retrains the offer prioritization model using feedback collected from users
# MAGIC and registers it as a new **challenger model** in Unity Catalog.
# MAGIC
# MAGIC ## Workflow
# MAGIC 1. Load feedback data from the feedback table
# MAGIC 2. Merge feedback signals with original training data
# MAGIC 3. Adjust target scores based on feedback (approved → boost, rejected → penalize)
# MAGIC 4. Retrain the model with enhanced data
# MAGIC 5. Compare with the current champion model
# MAGIC 6. Register as challenger in Unity Catalog

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
from typing import Dict, List, Tuple, Optional

# ML libraries
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import lightgbm as lgb

# MLflow
import mlflow
import mlflow.sklearn
import mlflow.lightgbm
from mlflow.models.signature import infer_signature
from mlflow.tracking import MlflowClient

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Unity Catalog configuration
CATALOG_NAME = "demos"
SCHEMA_NAME = "offer_prioritization"
MODEL_NAME = "healthcare_offer_prioritizer"

# MLflow experiment
EXPERIMENT_NAME = "/Shared/healthcare_offer_prioritization"

# Table names
FEEDBACK_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.offer_feedback"
RECOMMENDATIONS_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.member_offer_recommendations_with_reasoning"
FEATURES_TABLE = f"{CATALOG_NAME}.{SCHEMA_NAME}.member_features"

# Model registry path for Unity Catalog
UC_MODEL_PATH = f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}"

# Feedback impact settings
FEEDBACK_SETTINGS = {
    "approved_boost": 15.0,      # Increase score for approved offers
    "rejected_penalty": -20.0,   # Decrease score for rejected offers
    "min_feedback_count": 10,    # Minimum feedback records needed to retrain
    "feedback_weight": 0.3,      # Weight of feedback-adjusted samples in training
}

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
# MAGIC ## Load Required Modules

# COMMAND ----------

# Import required modules
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

# MAGIC %md
# MAGIC ## Load Feedback Data

# COMMAND ----------

def load_feedback_data() -> pd.DataFrame:
    """
    Load feedback data from the feedback table.
    
    Returns:
        DataFrame with feedback records
    """
    print(f"📥 Loading feedback from: {FEEDBACK_TABLE}")
    
    try:
        feedback_df = spark.sql(f"""
            SELECT 
                member_id,
                offer_id,
                feedback,
                feedback_text,
                feedback_time
            FROM {FEEDBACK_TABLE}
            WHERE feedback IN ('approved', 'rejected')
            ORDER BY feedback_time DESC
        """).toPandas()
        
        print(f"✅ Loaded {len(feedback_df):,} feedback records")
        return feedback_df
        
    except Exception as e:
        print(f"⚠️ Error loading feedback: {e}")
        return pd.DataFrame()

# COMMAND ----------

def analyze_feedback(feedback_df: pd.DataFrame) -> Dict:
    """
    Analyze feedback statistics.
    
    Args:
        feedback_df: Feedback DataFrame
        
    Returns:
        Dictionary with feedback statistics
    """
    if len(feedback_df) == 0:
        return {"total": 0, "approved": 0, "rejected": 0}
    
    stats = {
        "total": len(feedback_df),
        "approved": len(feedback_df[feedback_df['feedback'] == 'approved']),
        "rejected": len(feedback_df[feedback_df['feedback'] == 'rejected']),
        "unique_members": feedback_df['member_id'].nunique(),
        "unique_offers": feedback_df['offer_id'].nunique(),
    }
    
    stats["approval_rate"] = stats["approved"] / stats["total"] if stats["total"] > 0 else 0
    
    return stats

# COMMAND ----------

# Load and analyze feedback
feedback_df = load_feedback_data()
feedback_stats = analyze_feedback(feedback_df)

print("\n📊 Feedback Statistics:")
print("=" * 50)
for key, value in feedback_stats.items():
    if isinstance(value, float):
        print(f"   {key}: {value:.1%}" if key.endswith('rate') else f"   {key}: {value:.2f}")
    else:
        print(f"   {key}: {value:,}")

# COMMAND ----------

# Visualize feedback by offer
if len(feedback_df) > 0:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Feedback counts by offer
    ax1 = axes[0]
    offer_feedback = feedback_df.groupby(['offer_id', 'feedback']).size().unstack(fill_value=0)
    if 'approved' not in offer_feedback.columns:
        offer_feedback['approved'] = 0
    if 'rejected' not in offer_feedback.columns:
        offer_feedback['rejected'] = 0
    offer_feedback = offer_feedback[['approved', 'rejected']]
    offer_feedback.plot(kind='barh', stacked=True, ax=ax1, color=['#2ecc71', '#e74c3c'])
    ax1.set_xlabel('Feedback Count')
    ax1.set_title('Feedback by Offer', fontweight='bold')
    ax1.legend(['Approved', 'Rejected'])
    
    # Approval rate by offer
    ax2 = axes[1]
    approval_rates = feedback_df.groupby('offer_id').apply(
        lambda x: (x['feedback'] == 'approved').mean()
    ).sort_values()
    colors = ['#e74c3c' if r < 0.5 else '#2ecc71' for r in approval_rates.values]
    ax2.barh(approval_rates.index, approval_rates.values, color=colors)
    ax2.axvline(x=0.5, color='gray', linestyle='--', alpha=0.7)
    ax2.set_xlabel('Approval Rate')
    ax2.set_title('Approval Rate by Offer', fontweight='bold')
    ax2.set_xlim(0, 1)
    
    plt.tight_layout()
    plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Check Minimum Feedback Threshold

# COMMAND ----------

# Check if we have enough feedback to retrain
min_feedback = FEEDBACK_SETTINGS["min_feedback_count"]

if feedback_stats["total"] < min_feedback:
    print(f"⚠️ Insufficient feedback for retraining.")
    print(f"   Current: {feedback_stats['total']} records")
    print(f"   Required: {min_feedback} records")
    print(f"\n   Generating synthetic feedback for demo purposes...")
    
    # Generate synthetic feedback for demo
    catalog = OfferCatalog()
    synthetic_feedback = []
    
    # Generate 100 synthetic feedback records
    np.random.seed(42)
    for i in range(100):
        member_id = f"MEM_{np.random.randint(1, 1000):05d}"
        offer = np.random.choice(catalog.offers)
        feedback_type = np.random.choice(['approved', 'rejected'], p=[0.65, 0.35])
        
        synthetic_feedback.append({
            'member_id': member_id,
            'offer_id': offer['offer_id'],
            'feedback': feedback_type,
            'feedback_text': None,
            'feedback_time': datetime.now()
        })
    
    feedback_df = pd.DataFrame(synthetic_feedback)
    feedback_stats = analyze_feedback(feedback_df)
    
    print(f"\n✅ Generated {len(feedback_df)} synthetic feedback records")
    print(f"   Approval rate: {feedback_stats['approval_rate']:.1%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Base Training Data

# COMMAND ----------

# Generate synthetic data (in production, load from Delta tables)
print("🔄 Generating base training data...")

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

# Initialize offer catalog
catalog = OfferCatalog()

# Create base training data with rule-based scores
X_base, y_base = create_training_data(features_df, catalog)

print(f"\n✅ Base Training Data:")
print(f"   Samples: {len(X_base):,}")
print(f"   Features: {X_base.shape[1]}")
print(f"   Offers: {y_base.shape[1]}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Incorporate Feedback into Training Data

# COMMAND ----------

def create_feedback_adjusted_targets(
    features_df: pd.DataFrame,
    y_base: pd.DataFrame,
    feedback_df: pd.DataFrame,
    catalog: OfferCatalog,
    approved_boost: float = 15.0,
    rejected_penalty: float = -20.0
) -> pd.DataFrame:
    """
    Adjust target scores based on feedback.
    
    For each member-offer pair with feedback:
    - If approved: Boost the target score
    - If rejected: Penalize the target score
    
    Args:
        features_df: Member features with member_id
        y_base: Base target scores
        feedback_df: Feedback records
        catalog: Offer catalog
        approved_boost: Score increase for approved offers
        rejected_penalty: Score decrease for rejected offers
        
    Returns:
        Adjusted target scores DataFrame
    """
    print("🔧 Adjusting target scores based on feedback...")
    
    y_adjusted = y_base.copy()
    member_ids = features_df['member_id'].values
    offer_ids = catalog.get_offer_ids()
    
    adjustments_made = 0
    
    for _, row in feedback_df.iterrows():
        member_id = row['member_id']
        offer_id = row['offer_id']
        feedback = row['feedback']
        
        # Find member index
        member_mask = member_ids == member_id
        if not member_mask.any():
            continue
        
        # Find offer column
        offer_col = f"score_{offer_id}"
        if offer_col not in y_adjusted.columns:
            continue
        
        # Apply adjustment
        member_idx = np.where(member_mask)[0][0]
        current_score = y_adjusted.iloc[member_idx][offer_col]
        
        if feedback == 'approved':
            new_score = min(100, current_score + approved_boost)
        else:  # rejected
            new_score = max(0, current_score + rejected_penalty)
        
        y_adjusted.iloc[member_idx, y_adjusted.columns.get_loc(offer_col)] = new_score
        adjustments_made += 1
    
    print(f"✅ Made {adjustments_made:,} score adjustments based on feedback")
    
    return y_adjusted

# COMMAND ----------

def create_feedback_weighted_dataset(
    X_base: pd.DataFrame,
    y_adjusted: pd.DataFrame,
    feedback_df: pd.DataFrame,
    features_df: pd.DataFrame,
    feedback_weight: float = 0.3
) -> Tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    """
    Create a weighted training dataset that emphasizes feedback samples.
    
    Members with feedback are oversampled to ensure the model learns from feedback.
    
    Args:
        X_base: Base feature matrix
        y_adjusted: Feedback-adjusted targets
        feedback_df: Feedback records
        features_df: Member features with member_id
        feedback_weight: Proportion of training data that should be feedback samples
        
    Returns:
        Tuple of (X_weighted, y_weighted, sample_weights)
    """
    print("🔧 Creating weighted training dataset...")
    
    member_ids = features_df['member_id'].values
    feedback_member_ids = set(feedback_df['member_id'].unique())
    
    # Identify indices of members with feedback
    feedback_mask = np.array([mid in feedback_member_ids for mid in member_ids])
    feedback_indices = np.where(feedback_mask)[0]
    non_feedback_indices = np.where(~feedback_mask)[0]
    
    print(f"   Members with feedback: {len(feedback_indices):,}")
    print(f"   Members without feedback: {len(non_feedback_indices):,}")
    
    if len(feedback_indices) == 0:
        print("   ⚠️ No matching members found, using base data")
        return X_base, y_adjusted, np.ones(len(X_base))
    
    # Calculate how many times to oversample feedback members
    total_samples = len(X_base)
    target_feedback_samples = int(total_samples * feedback_weight)
    oversample_factor = max(1, target_feedback_samples // len(feedback_indices))
    
    print(f"   Oversampling feedback members by factor: {oversample_factor}x")
    
    # Create combined indices
    combined_indices = np.concatenate([
        non_feedback_indices,
        np.tile(feedback_indices, oversample_factor)
    ])
    np.random.shuffle(combined_indices)
    
    # Create weighted dataset
    X_weighted = X_base.iloc[combined_indices].reset_index(drop=True)
    y_weighted = y_adjusted.iloc[combined_indices].reset_index(drop=True)
    
    # Create sample weights (higher for feedback samples)
    sample_weights = np.ones(len(combined_indices))
    for i, idx in enumerate(combined_indices):
        if idx in feedback_indices:
            sample_weights[i] = 2.0  # Give feedback samples higher weight
    
    print(f"✅ Weighted dataset created: {len(X_weighted):,} samples")
    
    return X_weighted, y_weighted, sample_weights

# COMMAND ----------

# Adjust targets based on feedback
y_adjusted = create_feedback_adjusted_targets(
    features_df=features_df,
    y_base=y_base,
    feedback_df=feedback_df,
    catalog=catalog,
    approved_boost=FEEDBACK_SETTINGS["approved_boost"],
    rejected_penalty=FEEDBACK_SETTINGS["rejected_penalty"]
)

# Create weighted dataset
X_weighted, y_weighted, sample_weights = create_feedback_weighted_dataset(
    X_base=X_base,
    y_adjusted=y_adjusted,
    feedback_df=feedback_df,
    features_df=features_df,
    feedback_weight=FEEDBACK_SETTINGS["feedback_weight"]
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Train Challenger Model

# COMMAND ----------

# Split data
X_train, X_test, y_train, y_test, weights_train, weights_test = train_test_split(
    X_weighted, y_weighted, sample_weights,
    test_size=TEST_SIZE, 
    random_state=RANDOM_STATE
)

print(f"\n📊 Data Split:")
print(f"   Training: {len(X_train):,} samples")
print(f"   Testing: {len(X_test):,} samples")

# COMMAND ----------

# Set MLflow experiment
mlflow.set_experiment(EXPERIMENT_NAME)
print(f"✅ MLflow Experiment: {EXPERIMENT_NAME}")

# COMMAND ----------

def train_challenger_model(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.DataFrame,
    y_test: pd.DataFrame,
    weights_train: np.ndarray,
    model_params: Dict,
    feature_names: List[str],
    feedback_stats: Dict,
    catalog: OfferCatalog,
    run_name: str = "challenger_model"
) -> Tuple[OfferPrioritizationModel, str, Dict]:
    """
    Train and log challenger model to MLflow.
    
    Returns:
        Tuple of (trained_model, run_id, metrics)
    """
    
    with mlflow.start_run(run_name=run_name) as run:
        run_id = run.info.run_id
        print(f"\n🚀 MLflow Run Started: {run_id}")
        
        # Log parameters
        mlflow.log_params({
            "model_type": "LightGBM MultiOutput (Feedback-Enhanced)",
            "n_estimators": model_params.get("n_estimators", 200),
            "learning_rate": model_params.get("learning_rate", 0.05),
            "num_leaves": model_params.get("num_leaves", 31),
            "feature_fraction": model_params.get("feature_fraction", 0.8),
            "test_size": TEST_SIZE,
            "n_features": len(feature_names),
            "n_offers": y_train.shape[1],
            "n_training_samples": len(X_train),
            "feedback_count": feedback_stats["total"],
            "feedback_approval_rate": feedback_stats.get("approval_rate", 0),
            "approved_boost": FEEDBACK_SETTINGS["approved_boost"],
            "rejected_penalty": FEEDBACK_SETTINGS["rejected_penalty"],
            "feedback_weight": FEEDBACK_SETTINGS["feedback_weight"],
        })
        
        # Log feedback statistics
        mlflow.log_metrics({
            "feedback_total": feedback_stats["total"],
            "feedback_approved": feedback_stats.get("approved", 0),
            "feedback_rejected": feedback_stats.get("rejected", 0),
        })
        
        # Train model
        print("\n📈 Training challenger model with feedback-enhanced data...")
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
        importance_df.to_csv("/tmp/feature_importance_challenger.csv", index=False)
        mlflow.log_artifact("/tmp/feature_importance_challenger.csv")
        
        # Create feature importance plot
        plt.figure(figsize=(10, 8))
        top_features = importance_df.head(20)
        plt.barh(range(len(top_features)), top_features['importance'].values)
        plt.yticks(range(len(top_features)), top_features['feature'].values)
        plt.xlabel('Importance')
        plt.title('Top 20 Feature Importances (Challenger Model)')
        plt.tight_layout()
        plt.savefig("/tmp/feature_importance_challenger.png", dpi=150, bbox_inches='tight')
        mlflow.log_artifact("/tmp/feature_importance_challenger.png")
        plt.close()
        
        # Create model signature
        signature = infer_signature(X_train, raw_scores)
        
        # Log the model
        print("\n📦 Logging model to MLflow...")
        mlflow.sklearn.log_model(
            model.model,
            artifact_path="model",
            signature=signature,
            registered_model_name=None  # Will register separately as challenger
        )
        
        # Save full model wrapper
        import joblib
        model.save("/tmp/offer_model_challenger.joblib")
        mlflow.log_artifact("/tmp/offer_model_challenger.joblib")
        
        # Log metadata
        mlflow.log_dict({
            "offer_ids": catalog.get_offer_ids(),
            "feature_names": feature_names,
            "model_params": model_params,
            "feedback_settings": FEEDBACK_SETTINGS,
            "training_type": "feedback_enhanced"
        }, "model_metadata.json")
        
        # Log feedback summary
        mlflow.log_dict(feedback_stats, "feedback_stats.json")
        
        print(f"\n✅ MLflow Run Complete: {run_id}")
        
        return model, run_id, test_metrics

# COMMAND ----------

# Train challenger model
feature_names = list(X_weighted.columns)

challenger_model, challenger_run_id, challenger_metrics = train_challenger_model(
    X_train, X_test, y_train, y_test, weights_train,
    MODEL_PARAMS, feature_names, feedback_stats, catalog,
    run_name=f"challenger_feedback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Compare with Champion Model

# COMMAND ----------

def load_champion_model_metrics() -> Optional[Dict]:
    """
    Load metrics from the current champion model in Unity Catalog.
    
    Returns:
        Dictionary with champion metrics or None if not found
    """
    print(f"\n📦 Loading champion model metrics from: {UC_MODEL_PATH}")
    
    try:
        client = MlflowClient()
        
        # Try to get the champion version (alias)
        try:
            champion_version = client.get_model_version_by_alias(UC_MODEL_PATH, "champion")
            run_id = champion_version.run_id
            version = champion_version.version
            print(f"   Found champion version: {version} (alias)")
        except Exception:
            # Fallback to latest version
            versions = client.search_model_versions(f"name='{UC_MODEL_PATH}'")
            if not versions:
                print("   ⚠️ No model versions found")
                return None
            latest_version = max(versions, key=lambda v: int(v.version))
            run_id = latest_version.run_id
            version = latest_version.version
            print(f"   Using latest version: {version}")
        
        # Get metrics from the run
        run = client.get_run(run_id)
        metrics = run.data.metrics
        
        champion_metrics = {
            "version": version,
            "run_id": run_id,
            "test_rmse": metrics.get("test_rmse", 0),
            "test_mae": metrics.get("test_mae", 0),
            "test_r2": metrics.get("test_r2", 0),
        }
        
        print(f"   Champion Test RMSE: {champion_metrics['test_rmse']:.4f}")
        print(f"   Champion Test R²: {champion_metrics['test_r2']:.4f}")
        
        return champion_metrics
        
    except Exception as e:
        print(f"   ⚠️ Could not load champion metrics: {e}")
        return None

# COMMAND ----------

# Load champion metrics
champion_metrics = load_champion_model_metrics()

# Compare models
print("\n" + "=" * 60)
print("📊 MODEL COMPARISON")
print("=" * 60)

if champion_metrics:
    print(f"\n🏆 Champion Model (v{champion_metrics['version']}):")
    print(f"   Test RMSE: {champion_metrics['test_rmse']:.4f}")
    print(f"   Test R²: {champion_metrics['test_r2']:.4f}")
else:
    print("\n🏆 Champion Model: Not found (this will be the first version)")

print(f"\n🥊 Challenger Model (Feedback-Enhanced):")
print(f"   Test RMSE: {challenger_metrics['rmse']:.4f}")
print(f"   Test R²: {challenger_metrics['r2']:.4f}")
print(f"   Feedback records used: {feedback_stats['total']:,}")

# Determine if challenger is better
if champion_metrics:
    rmse_improvement = champion_metrics['test_rmse'] - challenger_metrics['rmse']
    r2_improvement = challenger_metrics['r2'] - champion_metrics['test_r2']
    
    print(f"\n📈 Improvement:")
    print(f"   RMSE: {rmse_improvement:+.4f} ({'better' if rmse_improvement > 0 else 'worse'})")
    print(f"   R²: {r2_improvement:+.4f} ({'better' if r2_improvement > 0 else 'worse'})")
    
    challenger_is_better = rmse_improvement > 0 or r2_improvement > 0.01
else:
    challenger_is_better = True

print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Register Challenger Model in Unity Catalog

# COMMAND ----------

def register_challenger_model(
    run_id: str,
    model_name: str,
    catalog_name: str,
    schema_name: str,
    description: str,
    feedback_stats: Dict,
    metrics: Dict
) -> Tuple[str, str]:
    """
    Register challenger model in Unity Catalog.
    
    Args:
        run_id: MLflow run ID
        model_name: Model name
        catalog_name: Unity Catalog name
        schema_name: Schema name
        description: Model description
        feedback_stats: Feedback statistics used for training
        metrics: Model performance metrics
        
    Returns:
        Tuple of (version, alias)
    """
    full_model_name = f"{catalog_name}.{schema_name}.{model_name}"
    model_uri = f"runs:/{run_id}/model"
    
    print(f"\n📝 Registering challenger model: {full_model_name}")
    
    # Register the model
    model_version = mlflow.register_model(
        model_uri=model_uri,
        name=full_model_name
    )
    
    version = model_version.version
    print(f"✅ Model registered: {full_model_name}")
    print(f"   Version: {version}")
    
    # Update model version description
    client = MlflowClient()
    
    version_description = f"""
Challenger model trained with user feedback.

**Training Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

**Feedback Statistics:**
- Total feedback records: {feedback_stats['total']:,}
- Approved: {feedback_stats.get('approved', 0):,}
- Rejected: {feedback_stats.get('rejected', 0):,}
- Approval rate: {feedback_stats.get('approval_rate', 0):.1%}

**Performance Metrics:**
- Test RMSE: {metrics['rmse']:.4f}
- Test MAE: {metrics['mae']:.4f}
- Test R²: {metrics['r2']:.4f}

**Feedback Settings:**
- Approved boost: {FEEDBACK_SETTINGS['approved_boost']}
- Rejected penalty: {FEEDBACK_SETTINGS['rejected_penalty']}
- Feedback weight: {FEEDBACK_SETTINGS['feedback_weight']}
"""
    
    client.update_model_version(
        name=full_model_name,
        version=version,
        description=version_description
    )
    
    # Set alias to "challenger"
    try:
        client.set_registered_model_alias(
            name=full_model_name,
            alias="challenger",
            version=version
        )
        print(f"   Alias set: @challenger → v{version}")
        alias = "challenger"
    except Exception as e:
        print(f"   ⚠️ Could not set alias: {e}")
        alias = None
    
    return version, alias

# COMMAND ----------

# Register challenger model
challenger_version, challenger_alias = register_challenger_model(
    run_id=challenger_run_id,
    model_name=MODEL_NAME,
    catalog_name=CATALOG_NAME,
    schema_name=SCHEMA_NAME,
    description="Challenger model trained with user feedback",
    feedback_stats=feedback_stats,
    metrics=challenger_metrics
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Promote Challenger to Champion (Optional)

# COMMAND ----------

def promote_challenger_to_champion(
    model_name: str,
    catalog_name: str,
    schema_name: str,
    challenger_version: str
) -> bool:
    """
    Promote challenger model to champion.
    
    This swaps the @champion alias to point to the challenger version.
    
    Args:
        model_name: Model name
        catalog_name: Catalog name
        schema_name: Schema name
        challenger_version: Version to promote
        
    Returns:
        True if successful
    """
    full_model_name = f"{catalog_name}.{schema_name}.{model_name}"
    
    print(f"\n🏆 Promoting challenger v{challenger_version} to champion...")
    
    try:
        client = MlflowClient()
        
        # Set champion alias to challenger version
        client.set_registered_model_alias(
            name=full_model_name,
            alias="champion",
            version=challenger_version
        )
        
        print(f"✅ Champion alias now points to v{challenger_version}")
        print(f"   Model: {full_model_name}@champion")
        
        return True
        
    except Exception as e:
        print(f"⚠️ Could not promote: {e}")
        return False

# COMMAND ----------

# MAGIC %md
# MAGIC ### Decision: Promote or Keep as Challenger
# MAGIC
# MAGIC Set `PROMOTE_TO_CHAMPION = True` to promote the challenger model to champion.

# COMMAND ----------

# ============================================================
# CONFIGURATION: Set to True to promote challenger to champion
# ============================================================
PROMOTE_TO_CHAMPION = False  # Set to True to promote automatically

if challenger_is_better and PROMOTE_TO_CHAMPION:
    promote_challenger_to_champion(
        MODEL_NAME, CATALOG_NAME, SCHEMA_NAME, challenger_version
    )
elif challenger_is_better:
    print("\n📋 Challenger model is better but not auto-promoted.")
    print("   To promote manually, run:")
    print(f"   promote_challenger_to_champion('{MODEL_NAME}', '{CATALOG_NAME}', '{SCHEMA_NAME}', '{challenger_version}')")
else:
    print("\n📋 Challenger model registered but not promoted (not better than champion).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Validation: Test Challenger Predictions

# COMMAND ----------

# Sample predictions from challenger
print("\n🔍 Sample Predictions from Challenger Model:")
print("=" * 60)

# Get predictions for first 5 test samples
sample_X = X_test.head(5)
sample_scores, top_offers = challenger_model.predict(sample_X, return_top_n=3)

# Display results
for i in range(5):
    member_offers = top_offers[top_offers['member_idx'] == i]
    print(f"\nSample {i+1}:")
    for _, row in member_offers.iterrows():
        print(f"  {row['rank']}. {row['offer_name']} (Score: {row['priority_score']:.1f})")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("\n" + "=" * 70)
print("📊 MODEL RETRAINING SUMMARY")
print("=" * 70)

print(f"\n📥 Feedback Data:")
print(f"   Total feedback records: {feedback_stats['total']:,}")
print(f"   Approved: {feedback_stats.get('approved', 0):,}")
print(f"   Rejected: {feedback_stats.get('rejected', 0):,}")
print(f"   Approval rate: {feedback_stats.get('approval_rate', 0):.1%}")

print(f"\n🎯 Training Configuration:")
print(f"   Approved boost: +{FEEDBACK_SETTINGS['approved_boost']}")
print(f"   Rejected penalty: {FEEDBACK_SETTINGS['rejected_penalty']}")
print(f"   Feedback weight: {FEEDBACK_SETTINGS['feedback_weight']:.0%}")

print(f"\n🥊 Challenger Model:")
print(f"   Run ID: {challenger_run_id}")
print(f"   Version: {challenger_version}")
print(f"   Alias: @{challenger_alias or 'none'}")
print(f"   Test RMSE: {challenger_metrics['rmse']:.4f}")
print(f"   Test R²: {challenger_metrics['r2']:.4f}")

if champion_metrics:
    print(f"\n🏆 vs Champion Model:")
    rmse_diff = champion_metrics['test_rmse'] - challenger_metrics['rmse']
    r2_diff = challenger_metrics['r2'] - champion_metrics['test_r2']
    print(f"   RMSE improvement: {rmse_diff:+.4f}")
    print(f"   R² improvement: {r2_diff:+.4f}")
    print(f"   Better: {'Yes ✅' if challenger_is_better else 'No ❌'}")

print(f"\n📦 Unity Catalog Model:")
print(f"   Path: {UC_MODEL_PATH}")
print(f"   Challenger: {UC_MODEL_PATH}@challenger (v{challenger_version})")

if PROMOTE_TO_CHAMPION and challenger_is_better:
    print(f"   Champion: {UC_MODEL_PATH}@champion (v{challenger_version}) ← PROMOTED")
else:
    print(f"   Champion: Unchanged")

print("\n" + "=" * 70)
print("✅ Model retraining complete!")
print("=" * 70)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Next Steps
# MAGIC
# MAGIC 1. **A/B Testing**: Route a percentage of traffic to the challenger model
# MAGIC 2. **Monitor Performance**: Track real-world metrics (click-through, conversions)
# MAGIC 3. **Collect More Feedback**: Continue gathering feedback for future retraining
# MAGIC 4. **Promote When Ready**: Use `promote_challenger_to_champion()` when confident
# MAGIC
# MAGIC ### Scheduled Retraining
# MAGIC
# MAGIC To automate retraining, create a Databricks Job that runs this notebook on a schedule:
# MAGIC
# MAGIC ```python
# MAGIC # Example: Weekly retraining job configuration
# MAGIC {
# MAGIC     "name": "Offer Model Retraining",
# MAGIC     "schedule": {
# MAGIC         "quartz_cron_expression": "0 0 2 ? * SUN *",  # Every Sunday at 2 AM
# MAGIC         "timezone_id": "America/New_York"
# MAGIC     },
# MAGIC     "tasks": [
# MAGIC         {
# MAGIC             "task_key": "retrain_model",
# MAGIC             "notebook_task": {
# MAGIC                 "notebook_path": "/Repos/project/notebooks/05_model_retraining"
# MAGIC             }
# MAGIC         }
# MAGIC     ]
# MAGIC }
# MAGIC ```

