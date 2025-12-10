# Databricks notebook source
# MAGIC %md
# MAGIC # 04 - Model Inference & Offer Recommendations
# MAGIC 
# MAGIC This notebook demonstrates how to use the trained model to generate
# MAGIC personalized offer recommendations for healthcare members.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

import sys
import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import Dict, List, Optional

import mlflow
import mlflow.sklearn

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Model configuration
MODEL_NAME = "healthcare_offer_prioritizer"
CATALOG_NAME = "healthcare_demo"
SCHEMA_NAME = "offer_prioritization"

# For Unity Catalog: f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}"
# For Workspace registry: MODEL_NAME

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Model from Registry

# COMMAND ----------

def load_model_from_registry(
    model_name: str,
    version: str = None,
    alias: str = None,
    catalog_name: str = None,
    schema_name: str = None
) -> mlflow.pyfunc.PyFuncModel:
    """
    Load model from MLflow Model Registry (supports both Unity Catalog and Workspace registry)
    
    Args:
        model_name: Registered model name
        version: Specific version number (e.g., "1", "2")
        alias: Model alias for Unity Catalog (e.g., "champion", "production")
        catalog_name: Unity Catalog name (if using UC)
        schema_name: Schema name (if using UC)
        
    Returns:
        Loaded MLflow model
    """
    # Build full model name for Unity Catalog
    if catalog_name and schema_name:
        full_model_name = f"{catalog_name}.{schema_name}.{model_name}"
    else:
        full_model_name = model_name
    
    # Build model URI based on what's provided
    if alias:
        # Unity Catalog uses aliases (e.g., @champion, @production)
        model_uri = f"models:/{full_model_name}@{alias}"
    elif version:
        # Specific version number
        model_uri = f"models:/{full_model_name}/{version}"
    else:
        # Default to version 1 if nothing specified
        model_uri = f"models:/{full_model_name}/1"
    
    print(f"📦 Loading model from: {model_uri}")
    
    try:
        model = mlflow.sklearn.load_model(model_uri)
        print("✅ Model loaded successfully")
        return model
    except Exception as e:
        print(f"⚠️ Failed to load model: {str(e)}")
        print("   Falling back to training a fresh model for demo...")
        return None

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
        create_training_data
    )
except ImportError:
    exec(open("../models/offer_model.py").read())

# COMMAND ----------

def wrap_loaded_model(
    sklearn_model,
    catalog: OfferCatalog,
    feature_names: List[str],
    scaler=None
) -> OfferPrioritizationModel:
    """
    Wrap a loaded sklearn model back into OfferPrioritizationModel.
    
    When loading from MLflow registry, you get the raw sklearn MultiOutputRegressor.
    This function wraps it back into our custom class with predict(return_top_n=...).
    
    Args:
        sklearn_model: The loaded sklearn MultiOutputRegressor
        catalog: OfferCatalog instance
        feature_names: List of feature names used during training
        scaler: Optional StandardScaler if features were scaled during training
        
    Returns:
        OfferPrioritizationModel wrapper
    """
    wrapper = OfferPrioritizationModel(catalog=catalog)
    wrapper.model = sklearn_model
    wrapper.feature_names = feature_names
    wrapper.scaler = scaler
    wrapper.is_fitted = True
    return wrapper

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configuration: Choose Model Source
# MAGIC 
# MAGIC Set `USE_REGISTRY_MODEL = True` to load from Unity Catalog, or `False` to train fresh.

# COMMAND ----------

# ============================================================
# CONFIGURATION: Set this to True to load from Unity Catalog
# ============================================================
USE_REGISTRY_MODEL = True

# Unity Catalog model location (update these to match your setup)
UC_MODEL_VERSION = "1"  # or use UC_MODEL_ALIAS instead
UC_MODEL_ALIAS = None   # e.g., "champion" or "production"

# Feature names from training (must match what was used in notebook 03)
# This will be loaded from the model metadata or you can specify manually
TRAINING_FEATURE_NAMES = None  # Will be auto-detected or loaded

# COMMAND ----------

# Initialize catalog
catalog = OfferCatalog()

if USE_REGISTRY_MODEL:
    # ============================================================
    # OPTION A: Load model from Unity Catalog
    # ============================================================
    print("📦 Loading model from Unity Catalog...")
    
    sklearn_model = load_model_from_registry(
        MODEL_NAME, 
        version=UC_MODEL_VERSION if not UC_MODEL_ALIAS else None,
        alias=UC_MODEL_ALIAS,
        catalog_name=CATALOG_NAME, 
        schema_name=SCHEMA_NAME
    )
    
    if sklearn_model is None:
        raise ValueError("Failed to load model from registry. Check model name and version/alias.")
    
    # Try to load the full model wrapper (with scaler) from the joblib artifact
    full_model_name = f"{CATALOG_NAME}.{SCHEMA_NAME}.{MODEL_NAME}"
    model = None
    feature_names = None
    
    try:
        import mlflow
        client = mlflow.tracking.MlflowClient()
        
        # Get the run ID from the model version
        if UC_MODEL_ALIAS:
            mv = client.get_model_version_by_alias(full_model_name, UC_MODEL_ALIAS)
        else:
            mv = client.get_model_version(full_model_name, UC_MODEL_VERSION)
        
        run_id = mv.run_id
        print(f"   ✓ Found model run: {run_id}")
        
        # Try to load the full model wrapper from joblib artifact (includes scaler)
        try:
            model_path = mlflow.artifacts.download_artifacts(
                run_id=run_id, 
                artifact_path="offer_model.joblib"
            )
            model = OfferPrioritizationModel.load(model_path)
            model.catalog = catalog  # Update catalog in case offers changed
            print(f"   ✓ Loaded full model wrapper with scaler from joblib artifact")
            feature_names = model.feature_names
        except Exception as e:
            print(f"   ⚠️ Could not load joblib artifact: {e}")
            print(f"   → Will wrap the sklearn model instead")
        
        # If joblib load failed, try to load metadata for feature names
        if model is None:
            try:
                metadata_path = mlflow.artifacts.download_artifacts(
                    run_id=run_id, 
                    artifact_path="model_metadata.json"
                )
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                feature_names = metadata.get('feature_names', [])
                print(f"   ✓ Loaded {len(feature_names)} feature names from model metadata")
            except Exception as e:
                print(f"   ⚠️ Could not load metadata: {e}")
                feature_names = TRAINING_FEATURE_NAMES
                
    except Exception as e:
        print(f"   ⚠️ Could not access model artifacts: {e}")
        feature_names = TRAINING_FEATURE_NAMES
    
    # Generate data for inference (we still need features for the members)
    print("\n🔄 Generating member data for inference...")
    members_df, claims_df, benefits_df, engagement_df = generate_all_data(
        n_members=10000,
        seed=123
    )
    
    features_df, feature_engineer = create_member_features(
        members_df, claims_df, benefits_df, engagement_df
    )
    
    # If we couldn't load the full model, wrap the sklearn model
    if model is None:
        # Use feature names from the generated features if not loaded from metadata
        if not feature_names:
            feature_names = [c for c in features_df.columns if c != 'member_id']
            print(f"   ✓ Using {len(feature_names)} feature names from generated features")
        
        # Wrap the loaded sklearn model
        model = wrap_loaded_model(
            sklearn_model=sklearn_model,
            catalog=catalog,
            feature_names=feature_names,
            scaler=None  # Note: predictions may differ slightly without the original scaler
        )
        print(f"   ⚠️ Model wrapped without original scaler - predictions may differ slightly")
    
    print(f"\n✅ Model loaded from Unity Catalog and ready for inference")

else:
    # ============================================================
    # OPTION B: Train a fresh model (for demo/development)
    # ============================================================
    print("🔄 Generating data and training fresh model for demo...")
    members_df, claims_df, benefits_df, engagement_df = generate_all_data(
        n_members=10000,
        seed=123
    )
    
    features_df, feature_engineer = create_member_features(
        members_df, claims_df, benefits_df, engagement_df
    )
    
    X, y = create_training_data(features_df, catalog)
    model = OfferPrioritizationModel(catalog=catalog)
    model.fit(X, y, feature_names=list(X.columns))
    
    print(f"\n✅ Fresh model trained and ready for inference")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Batch Inference Pipeline

# COMMAND ----------

class OfferRecommendationEngine:
    """
    Engine for generating personalized offer recommendations
    """
    
    def __init__(
        self,
        model: OfferPrioritizationModel,
        catalog: OfferCatalog = None
    ):
        self.model = model
        self.catalog = catalog or OfferCatalog()
        
    def get_recommendations(
        self,
        features_df: pd.DataFrame,
        top_n: int = 5,
        min_score: float = 30.0,
        exclude_categories: List[str] = None
    ) -> pd.DataFrame:
        """
        Generate offer recommendations for members
        
        Args:
            features_df: Member features with member_id
            top_n: Number of top offers per member
            min_score: Minimum score threshold
            exclude_categories: Categories to exclude
            
        Returns:
            DataFrame with recommendations
        """
        # Separate member_id from features
        member_ids = features_df['member_id'].values
        X = features_df.drop(columns=['member_id'])
        
        # Get predictions
        raw_scores, top_offers = self.model.predict(X, return_top_n=top_n * 2)  # Get extra for filtering
        
        # Add member_id
        top_offers['member_id'] = top_offers['member_idx'].map(
            lambda x: member_ids[x]
        )
        
        # Apply filters
        if min_score:
            top_offers = top_offers[top_offers['priority_score'] >= min_score]
        
        if exclude_categories:
            top_offers = top_offers[~top_offers['category'].isin(exclude_categories)]
        
        # Re-rank and limit to top_n
        top_offers['rank'] = top_offers.groupby('member_id').cumcount() + 1
        top_offers = top_offers[top_offers['rank'] <= top_n]
        
        # Add offer details
        top_offers['description'] = top_offers['offer_id'].apply(
            lambda x: self.catalog.get_offer_by_id(x).get('description', '') 
            if self.catalog.get_offer_by_id(x) else ''
        )
        
        # Reorder columns
        columns = ['member_id', 'rank', 'offer_id', 'offer_name', 'category', 
                   'priority_score', 'description']
        top_offers = top_offers[[c for c in columns if c in top_offers.columns]]
        
        return top_offers
    
    def get_member_profile_with_offers(
        self,
        member_id: str,
        features_df: pd.DataFrame,
        members_df: pd.DataFrame = None
    ) -> Dict:
        """
        Get detailed member profile with offer recommendations
        """
        # Get member features
        member_features = features_df[features_df['member_id'] == member_id]
        
        if len(member_features) == 0:
            return {"error": f"Member {member_id} not found"}
        
        # Get recommendations
        recommendations = self.get_recommendations(member_features, top_n=5)
        
        # Build profile
        profile = {
            "member_id": member_id,
            "features": {
                "age": int(member_features['age'].values[0]) if 'age' in member_features else None,
                "risk_score": float(member_features['risk_score'].values[0]) if 'risk_score' in member_features else None,
                "chronic_conditions": int(member_features['chronic_condition_count'].values[0]) if 'chronic_condition_count' in member_features else 0,
                "total_claims": int(member_features['total_claims_count'].values[0]) if 'total_claims_count' in member_features else 0,
                "total_engagements": int(member_features['total_engagements'].values[0]) if 'total_engagements' in member_features else 0,
            },
            "recommendations": recommendations.to_dict('records')
        }
        
        return profile

# COMMAND ----------

# Initialize recommendation engine
engine = OfferRecommendationEngine(model, catalog)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Batch Recommendations

# COMMAND ----------

# Generate recommendations for all members
print("🎯 Generating offer recommendations...")
all_recommendations = engine.get_recommendations(
    features_df=features_df,
    top_n=5,
    min_score=25.0
)

print(f"\n✅ Generated {len(all_recommendations):,} recommendations")
print(f"   for {all_recommendations['member_id'].nunique():,} members")

# COMMAND ----------

# Preview recommendations
all_recommendations.head(20)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Recommendation Analysis

# COMMAND ----------

# Offer distribution
offer_counts = all_recommendations.groupby('offer_name').agg({
    'member_id': 'count',
    'priority_score': 'mean'
}).reset_index()
offer_counts.columns = ['Offer', 'Recommendations', 'Avg Score']
offer_counts = offer_counts.sort_values('Recommendations', ascending=False)

print("\n📊 Offer Distribution:")
print(offer_counts.to_string(index=False))

# COMMAND ----------

# Visualize offer distribution
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Recommendation count by offer
ax1 = axes[0]
top_offers_chart = offer_counts.head(10)
ax1.barh(top_offers_chart['Offer'], top_offers_chart['Recommendations'], color='steelblue')
ax1.set_xlabel('Number of Recommendations')
ax1.set_title('Top 10 Most Recommended Offers', fontweight='bold')
ax1.invert_yaxis()

# Score distribution by category
ax2 = axes[1]
category_scores = all_recommendations.groupby('category')['priority_score'].mean().sort_values()
colors = plt.cm.Set3(np.linspace(0, 1, len(category_scores)))
ax2.barh(category_scores.index, category_scores.values, color=colors)
ax2.set_xlabel('Average Priority Score')
ax2.set_title('Average Score by Offer Category', fontweight='bold')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Individual Member Recommendations

# COMMAND ----------

# Get sample member profiles
sample_members = features_df['member_id'].sample(5, random_state=42).tolist()

for member_id in sample_members:
    profile = engine.get_member_profile_with_offers(member_id, features_df)
    
    print("\n" + "=" * 70)
    print(f"👤 MEMBER: {profile['member_id']}")
    print("=" * 70)
    print(f"\n📋 Profile:")
    print(f"   Age: {profile['features']['age']}")
    print(f"   Risk Score: {profile['features']['risk_score']:.1f}")
    print(f"   Chronic Conditions: {profile['features']['chronic_conditions']}")
    print(f"   Total Claims: {profile['features']['total_claims']}")
    print(f"   Total Engagements: {profile['features']['total_engagements']}")
    
    print(f"\n🎯 Top Offer Recommendations:")
    for rec in profile['recommendations']:
        print(f"\n   {rec['rank']}. {rec['offer_name']}")
        print(f"      Category: {rec['category']}")
        print(f"      Priority Score: {rec['priority_score']:.1f}")
        print(f"      {rec['description'][:60]}...")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Score Distribution Analysis

# COMMAND ----------

# Analyze score distributions by rank
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Score distribution by rank
ax1 = axes[0]
for rank in range(1, 6):
    rank_scores = all_recommendations[all_recommendations['rank'] == rank]['priority_score']
    ax1.hist(rank_scores, bins=30, alpha=0.6, label=f'Rank {rank}')
ax1.set_xlabel('Priority Score')
ax1.set_ylabel('Count')
ax1.set_title('Score Distribution by Rank', fontweight='bold')
ax1.legend()

# Box plot by category
ax2 = axes[1]
categories = all_recommendations['category'].unique()
data = [all_recommendations[all_recommendations['category'] == cat]['priority_score'] 
        for cat in categories]
bp = ax2.boxplot(data, labels=[c.replace('_', '\n') for c in categories], patch_artist=True)
for patch, color in zip(bp['boxes'], plt.cm.Set3(np.linspace(0, 1, len(categories)))):
    patch.set_facecolor(color)
ax2.set_ylabel('Priority Score')
ax2.set_title('Score Distribution by Category', fontweight='bold')
ax2.tick_params(axis='x', rotation=45)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Real-time Inference Example

# COMMAND ----------

def get_realtime_recommendations(
    member_data: Dict,
    model: OfferPrioritizationModel,
    feature_engineer,
    top_n: int = 5
) -> List[Dict]:
    """
    Get recommendations for a single member in real-time
    
    Args:
        member_data: Dictionary with member information
        model: Trained model
        feature_engineer: Feature engineer with encoders
        top_n: Number of recommendations
        
    Returns:
        List of offer recommendations
    """
    # In production, this would:
    # 1. Fetch member data from database
    # 2. Compute features using feature_engineer
    # 3. Call model.predict()
    
    # For demo, use random features
    import random
    
    feature_values = {
        'age': member_data.get('age', 45),
        'tenure_months': member_data.get('tenure_months', 24),
        'risk_score': member_data.get('risk_score', 30),
        'chronic_condition_count': member_data.get('chronic_conditions', 0),
        'total_claims_count': member_data.get('claims_count', 5),
        'total_engagements': member_data.get('engagements', 10),
        # Add other features with defaults...
    }
    
    # Create feature vector (simplified for demo)
    # In production, use full feature engineering pipeline
    
    print(f"\n⚡ Real-time recommendation for member:")
    print(f"   Age: {feature_values['age']}")
    print(f"   Risk Score: {feature_values['risk_score']}")
    print(f"   Chronic Conditions: {feature_values['chronic_condition_count']}")
    
    return {"status": "Demo mode - use full pipeline for production"}

# Example call
result = get_realtime_recommendations(
    member_data={
        "age": 55,
        "tenure_months": 36,
        "risk_score": 65,
        "chronic_conditions": 2,
        "claims_count": 12
    },
    model=model,
    feature_engineer=feature_engineer
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Recommendations to Delta Table

# COMMAND ----------

# # Save recommendations to Delta table
# recommendations_spark = spark.createDataFrame(all_recommendations)
# recommendations_spark.write.mode("overwrite").saveAsTable(
#     f"{CATALOG_NAME}.{SCHEMA_NAME}.offer_recommendations"
# )
# print("✓ Saved recommendations to Delta table")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export for Marketing Campaigns

# COMMAND ----------

# Create campaign-ready export
campaign_export = all_recommendations.copy()
campaign_export['recommendation_date'] = datetime.now().strftime('%Y-%m-%d')
campaign_export['campaign_id'] = 'HEALTH_OFFERS_Q4_2024'

# Aggregate by member for email campaigns
member_offers = campaign_export.groupby('member_id').apply(
    lambda x: ', '.join(x.sort_values('rank')['offer_name'].head(3))
).reset_index()
member_offers.columns = ['member_id', 'top_3_offers']

print("\n📧 Campaign Export Sample:")
print(member_offers.head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

print("\n" + "=" * 60)
print("📊 INFERENCE SUMMARY")
print("=" * 60)
print(f"\n👥 Members processed: {features_df['member_id'].nunique():,}")
print(f"🎯 Total recommendations: {len(all_recommendations):,}")
print(f"📋 Offers in catalog: {len(catalog.offers)}")
print(f"\n📈 Recommendation Stats:")
print(f"   Avg recommendations per member: {len(all_recommendations) / features_df['member_id'].nunique():.1f}")
print(f"   Avg priority score: {all_recommendations['priority_score'].mean():.1f}")
print(f"   Score range: {all_recommendations['priority_score'].min():.1f} - {all_recommendations['priority_score'].max():.1f}")
print(f"\n🏆 Top Recommended Offers:")
top_3 = offer_counts.head(3)
for _, row in top_3.iterrows():
    print(f"   • {row['Offer']}: {row['Recommendations']:,} recommendations")
print("\n" + "=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Per-Member Feature Importance (SHAP Explanations)
# MAGIC 
# MAGIC This section computes feature importance for each member's recommendations,
# MAGIC enabling an LLM to explain why specific offers were selected.

# COMMAND ----------

import shap

def compute_member_explanations(
    model: OfferPrioritizationModel,
    features_df: pd.DataFrame,
    recommendations_df: pd.DataFrame,
    top_features: int = 5
) -> pd.DataFrame:
    """
    Compute per-member feature importance for their recommended offers using SHAP.
    
    Args:
        model: Trained OfferPrioritizationModel
        features_df: Member features with member_id
        recommendations_df: Recommendations from get_recommendations()
        top_features: Number of top contributing features to return per offer
        
    Returns:
        DataFrame with member_id, offer_id, and feature explanations
    """
    # Prepare feature matrix
    member_ids = features_df['member_id'].values
    X = features_df.drop(columns=['member_id'])
    
    # Scale features if model uses scaling
    if model.scaler:
        X_scaled = model.scaler.transform(X)
    else:
        X_scaled = X.values
    
    # Get unique members with recommendations
    members_with_recs = recommendations_df['member_id'].unique()
    
    # Create SHAP explainers for each offer model (estimator)
    print("🔍 Computing SHAP explanations...")
    explainers = []
    shap_values_all = []
    
    for i, estimator in enumerate(model.model.estimators_):
        explainer = shap.TreeExplainer(estimator)
        shap_values = explainer.shap_values(X_scaled)
        shap_values_all.append(shap_values)
        explainers.append(explainer)
        if (i + 1) % 4 == 0:
            print(f"   Processed {i + 1}/{len(model.model.estimators_)} offer models...")
    
    print(f"   ✓ SHAP values computed for all {len(model.model.estimators_)} offers")
    
    # Build explanations for each member's recommendations
    explanations = []
    feature_names = model.feature_names
    
    for _, rec in recommendations_df.iterrows():
        member_id = rec['member_id']
        offer_id = rec['offer_id']
        offer_name = rec['offer_name']
        
        # Find member index
        member_idx = np.where(member_ids == member_id)[0]
        if len(member_idx) == 0:
            continue
        member_idx = member_idx[0]
        
        # Find offer index
        offer_idx = model.offer_ids.index(offer_id)
        
        # Get SHAP values for this member and offer
        member_shap = shap_values_all[offer_idx][member_idx]
        
        # Get top contributing features (by absolute SHAP value)
        top_indices = np.argsort(np.abs(member_shap))[::-1][:top_features]
        
        # Build feature contribution list
        feature_contributions = []
        for idx in top_indices:
            feature_name = feature_names[idx]
            shap_value = member_shap[idx]
            feature_value = X.iloc[member_idx, idx]
            
            # Determine direction of impact
            direction = "increases" if shap_value > 0 else "decreases"
            
            feature_contributions.append({
                "feature": feature_name,
                "value": round(float(feature_value), 2),
                "shap_value": round(float(shap_value), 3),
                "direction": direction
            })
        
        explanations.append({
            "member_id": member_id,
            "offer_id": offer_id,
            "offer_name": offer_name,
            "priority_score": rec['priority_score'],
            "rank": rec['rank'],
            "top_features": feature_contributions
        })
    
    return pd.DataFrame(explanations)

# COMMAND ----------

# Compute explanations for sample members
sample_member_ids = features_df['member_id'].sample(10, random_state=42).tolist()
sample_recs = all_recommendations[all_recommendations['member_id'].isin(sample_member_ids)]

explanations_df = compute_member_explanations(
    model=model,
    features_df=features_df,
    recommendations_df=sample_recs,
    top_features=5
)

print(f"\n✅ Generated explanations for {len(explanations_df)} member-offer pairs")

# COMMAND ----------

# MAGIC %md
# MAGIC ### View Explanations for Individual Members

# COMMAND ----------

def format_member_explanation_for_llm(member_id: str, explanations_df: pd.DataFrame, features_df: pd.DataFrame) -> str:
    """
    Format member explanations as structured text for LLM consumption.
    
    Returns a formatted string that an LLM can use to generate natural language summaries.
    """
    member_explanations = explanations_df[explanations_df['member_id'] == member_id]
    member_features = features_df[features_df['member_id'] == member_id].iloc[0]
    
    output = []
    output.append(f"=== MEMBER PROFILE: {member_id} ===")
    output.append(f"\nKey Demographics:")
    output.append(f"  - Age: {member_features.get('age', 'N/A')}")
    output.append(f"  - Risk Score: {member_features.get('risk_score', 'N/A'):.1f}")
    output.append(f"  - Chronic Conditions: {member_features.get('chronic_condition_count', 0)}")
    output.append(f"  - Total Claims: {member_features.get('total_claims_count', 0):.0f}")
    output.append(f"  - Total Engagements: {member_features.get('total_engagements', 0):.0f}")
    
    # Add condition flags
    conditions = []
    if member_features.get('has_diabetes', 0) == 1:
        conditions.append("Diabetes")
    if member_features.get('has_cardiovascular', 0) == 1:
        conditions.append("Cardiovascular")
    if member_features.get('has_respiratory', 0) == 1:
        conditions.append("Respiratory")
    if member_features.get('has_mental_health', 0) == 1:
        conditions.append("Mental Health")
    if conditions:
        output.append(f"  - Conditions: {', '.join(conditions)}")
    
    output.append(f"\n=== RECOMMENDED OFFERS ===")
    
    for _, row in member_explanations.iterrows():
        output.append(f"\n[Rank {row['rank']}] {row['offer_name']}")
        output.append(f"  Offer ID: {row['offer_id']}")
        output.append(f"  Priority Score: {row['priority_score']:.1f}/100")
        output.append(f"  \n  Top Factors Influencing This Recommendation:")
        
        for i, feat in enumerate(row['top_features'], 1):
            direction_symbol = "↑" if feat['direction'] == "increases" else "↓"
            output.append(
                f"    {i}. {feat['feature']}: {feat['value']} "
                f"({direction_symbol} score by {abs(feat['shap_value']):.2f})"
            )
    
    output.append("\n" + "=" * 50)
    return "\n".join(output)

# COMMAND ----------

# Display formatted explanations for sample members
print("\n" + "=" * 70)
print("📋 MEMBER EXPLANATIONS FOR LLM SUMMARIZATION")
print("=" * 70)

for member_id in sample_member_ids[:3]:  # Show first 3 members
    explanation_text = format_member_explanation_for_llm(member_id, explanations_df, features_df)
    print(explanation_text)
    print("\n")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Export Explanations as JSON for LLM Processing

# COMMAND ----------

def export_explanations_for_llm(explanations_df: pd.DataFrame, features_df: pd.DataFrame) -> List[Dict]:
    """
    Export explanations as structured JSON for LLM API consumption.
    """
    member_ids = explanations_df['member_id'].unique()
    
    all_member_data = []
    
    for member_id in member_ids:
        member_explanations = explanations_df[explanations_df['member_id'] == member_id]
        member_features = features_df[features_df['member_id'] == member_id].iloc[0]
        
        member_data = {
            "member_id": member_id,
            "profile": {
                "age": int(member_features.get('age', 0)),
                "risk_score": float(member_features.get('risk_score', 0)),
                "chronic_condition_count": int(member_features.get('chronic_condition_count', 0)),
                "total_claims": int(member_features.get('total_claims_count', 0)),
                "total_engagements": int(member_features.get('total_engagements', 0)),
                "is_senior": bool(member_features.get('is_senior', 0)),
                "has_diabetes": bool(member_features.get('has_diabetes', 0)),
                "has_cardiovascular": bool(member_features.get('has_cardiovascular', 0)),
                "has_respiratory": bool(member_features.get('has_respiratory', 0)),
                "has_mental_health": bool(member_features.get('has_mental_health', 0)),
                "is_complex_patient": bool(member_features.get('is_complex_patient', 0)),
            },
            "recommendations": []
        }
        
        for _, row in member_explanations.iterrows():
            rec_data = {
                "rank": int(row['rank']),
                "offer_id": row['offer_id'],
                "offer_name": row['offer_name'],
                "priority_score": float(row['priority_score']),
                "key_factors": row['top_features']
            }
            member_data["recommendations"].append(rec_data)
        
        all_member_data.append(member_data)
    
    return all_member_data

# COMMAND ----------

# Export to JSON format
llm_export = export_explanations_for_llm(explanations_df, features_df)

# Display sample JSON structure
print("\n📄 Sample JSON Export for LLM:")
print("=" * 50)
print(json.dumps(llm_export[0], indent=2))

# COMMAND ----------

# Save full export to file (for LLM processing)
with open("/tmp/member_explanations_for_llm.json", "w") as f:
    json.dump(llm_export, f, indent=2)

print(f"\n✅ Exported {len(llm_export)} member explanations to /tmp/member_explanations_for_llm.json")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Example LLM Prompt Template

# COMMAND ----------

llm_prompt_template = '''
You are a healthcare benefits advisor. Based on the member profile and recommended offers below,
write a personalized summary explaining why each offer is relevant for this member.

Use a warm, supportive tone. Focus on how each offer addresses the member's specific health needs
and circumstances. Keep explanations concise (2-3 sentences per offer).

MEMBER DATA:
{member_json}

Please write a personalized summary for this member explaining their top recommended offers
and why each one is particularly relevant for them.
'''

# Example usage
print("\n📝 Example LLM Prompt:")
print("=" * 50)
sample_prompt = llm_prompt_template.format(member_json=json.dumps(llm_export[0], indent=2))
print(sample_prompt[:2000] + "..." if len(sample_prompt) > 2000 else sample_prompt)

# COMMAND ----------

# MAGIC %md
# MAGIC ## LLM-Generated Offer Reasoning
# MAGIC 
# MAGIC This section uses an LLM to generate personalized 3-4 sentence explanations
# MAGIC for why each offer was recommended to a member based on feature importance.

# COMMAND ----------

from openai import OpenAI
import os

def get_llm_client():
    """
    Initialize LLM client for Databricks Foundation Model API.
    
    In Databricks, this uses the built-in Foundation Model API.
    For other environments, configure with your API endpoint and key.
    """
    
    # Try Databricks Foundation Model API first
    try:
        # Get workspace URL and token from Databricks context
        workspace_url = spark.conf.get("spark.databricks.workspaceUrl")
        api_token = dbutils.notebook.entry_point.getDbutils().notebook().getContext().apiToken().get()
        
        # Ensure workspace URL has https:// prefix
        if not workspace_url.startswith("https://"):
            workspace_url = f"https://{workspace_url}"
        
        # Construct the base URL for Foundation Model API
        base_url = f"{workspace_url}/serving-endpoints"
        
        client = OpenAI(
            api_key=api_token,
            base_url=base_url
        )
        
        # Test the connection with a simple check
        print(f"   Using Databricks Foundation Model API at: {base_url}")
        return client, "databricks-meta-llama-3-1-70b-instruct"
        
    except Exception as e:
        print(f"   ⚠️ Databricks Foundation Model API not available: {e}")
    
    # Try using Databricks SDK (alternative method)
    try:
        from databricks.sdk import WorkspaceClient
        from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
        
        print("   Trying Databricks SDK...")
        w = WorkspaceClient()
        # Return a wrapper that uses the SDK
        return ("databricks_sdk", w), "databricks-meta-llama-3-1-70b-instruct"
        
    except Exception as e:
        print(f"   ⚠️ Databricks SDK not available: {e}")
    
    # Fallback to environment variables (OpenAI or external API)
    api_key = os.environ.get("OPENAI_API_KEY") or os.environ.get("DATABRICKS_TOKEN")
    base_url = os.environ.get("OPENAI_BASE_URL", "https://api.openai.com/v1")
    
    if not api_key:
        print("   ⚠️ No API key found. Set OPENAI_API_KEY or DATABRICKS_TOKEN environment variable.")
        return None, None
    
    print(f"   Using API at: {base_url}")
    client = OpenAI(api_key=api_key, base_url=base_url)
    model = "gpt-4o-mini" if "openai.com" in base_url else "databricks-meta-llama-3-1-70b-instruct"
    return client, model

# COMMAND ----------

def generate_offer_reasoning(
    member_data: Dict,
    offer_data: Dict,
    client,
    model: str
) -> str:
    """
    Generate a 3-4 sentence reasoning for why a specific offer was recommended.
    
    Args:
        member_data: Member profile and context
        offer_data: Offer details with key contributing factors
        client: OpenAI-compatible client or Databricks SDK tuple
        model: Model name to use
        
    Returns:
        Generated reasoning text
    """
    # Format the key factors into readable text
    factors_text = []
    for factor in offer_data['key_factors']:
        direction = "positively" if factor['direction'] == "increases" else "negatively"
        factors_text.append(
            f"- {factor['feature'].replace('_', ' ')}: {factor['value']} "
            f"({direction} influences recommendation)"
        )
    
    prompt = f"""You are a healthcare benefits advisor writing personalized explanations for offer recommendations.

Based on the member profile and the key factors that influenced this recommendation, write a 3-4 sentence 
explanation for why this offer is being recommended. Be specific about how the member's characteristics 
make this offer relevant. Use a warm, supportive, and professional tone.

MEMBER PROFILE:
- Age: {member_data['profile']['age']}
- Risk Score: {member_data['profile']['risk_score']:.1f}
- Chronic Conditions: {member_data['profile']['chronic_condition_count']}
- Total Claims: {member_data['profile']['total_claims']}
- Has Diabetes: {member_data['profile']['has_diabetes']}
- Has Cardiovascular Issues: {member_data['profile']['has_cardiovascular']}
- Has Respiratory Issues: {member_data['profile']['has_respiratory']}
- Has Mental Health History: {member_data['profile']['has_mental_health']}
- Is Complex Patient: {member_data['profile']['is_complex_patient']}

RECOMMENDED OFFER:
- Offer Name: {offer_data['offer_name']}
- Priority Score: {offer_data['priority_score']:.1f}/100

KEY FACTORS INFLUENCING THIS RECOMMENDATION:
{chr(10).join(factors_text)}

Write a 3-4 sentence personalized explanation for why this offer is recommended for this member. 
Do not use bullet points. Do not mention scores or technical details. Focus on the member's health needs."""

    try:
        # Check if using Databricks SDK
        if isinstance(client, tuple) and client[0] == "databricks_sdk":
            from databricks.sdk.service.serving import ChatMessage, ChatMessageRole
            
            workspace_client = client[1]
            response = workspace_client.serving_endpoints.query(
                name=model,
                messages=[
                    ChatMessage(role=ChatMessageRole.SYSTEM, content="You are a helpful healthcare benefits advisor."),
                    ChatMessage(role=ChatMessageRole.USER, content=prompt)
                ],
                max_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
        else:
            # Use OpenAI-compatible client
            response = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": "You are a helpful healthcare benefits advisor."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=200,
                temperature=0.7
            )
            return response.choices[0].message.content.strip()
    except Exception as e:
        return f"[Error generating reasoning: {str(e)}]"

# COMMAND ----------

def generate_all_offer_reasonings(
    llm_export: List[Dict],
    client: OpenAI,
    model: str,
    max_members: int = None
) -> List[Dict]:
    """
    Generate LLM reasoning for all member-offer pairs.
    
    Args:
        llm_export: List of member data with recommendations
        client: OpenAI-compatible client
        model: Model name
        max_members: Limit number of members to process (for testing)
        
    Returns:
        Updated list with 'reasoning' field added to each recommendation
    """
    results = []
    members_to_process = llm_export[:max_members] if max_members else llm_export
    
    print(f"🤖 Generating LLM reasoning for {len(members_to_process)} members...")
    
    for i, member_data in enumerate(members_to_process):
        member_result = member_data.copy()
        member_result['recommendations'] = []
        
        for offer_data in member_data['recommendations']:
            offer_result = offer_data.copy()
            
            # Generate reasoning
            reasoning = generate_offer_reasoning(member_data, offer_data, client, model)
            offer_result['reasoning'] = reasoning
            
            member_result['recommendations'].append(offer_result)
        
        results.append(member_result)
        
        if (i + 1) % 5 == 0:
            print(f"   Processed {i + 1}/{len(members_to_process)} members...")
    
    print(f"✅ Generated reasoning for all {len(results)} members")
    return results

# COMMAND ----------

# MAGIC %md
# MAGIC ### LLM Configuration
# MAGIC 
# MAGIC Configure the LLM endpoint below. Options:
# MAGIC 1. **Databricks Foundation Model API** (default) - uses workspace token automatically
# MAGIC 2. **Custom Model Serving Endpoint** - set `LLM_ENDPOINT_NAME` to your endpoint name
# MAGIC 3. **External API** - set environment variables `OPENAI_API_KEY` and optionally `OPENAI_BASE_URL`

# COMMAND ----------

# LLM Configuration - update these as needed
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-1-70b-instruct"  # Change to your endpoint name if using custom serving

# Alternative: Use a specific Databricks Model Serving endpoint
# LLM_ENDPOINT_NAME = "your-custom-llm-endpoint"

# COMMAND ----------

# Initialize LLM client with detailed logging
print("🤖 Initializing LLM client...")
llm_client, llm_model = get_llm_client()

# Override model name if custom endpoint specified
if llm_client and LLM_ENDPOINT_NAME != "databricks-meta-llama-3-1-70b-instruct":
    llm_model = LLM_ENDPOINT_NAME
    print(f"   Using custom endpoint: {llm_model}")

if llm_client:
    print(f"✅ LLM client initialized with model: {llm_model}")
    
    # Test the connection
    print("\n🔍 Testing LLM connection...")
    try:
        test_response = generate_offer_reasoning(
            member_data={
                "member_id": "TEST",
                "profile": {
                    "age": 45, "risk_score": 30.0, "chronic_condition_count": 0,
                    "total_claims": 5, "total_engagements": 10,
                    "has_diabetes": False, "has_cardiovascular": False,
                    "has_respiratory": False, "has_mental_health": False,
                    "is_complex_patient": False
                }
            },
            offer_data={
                "offer_name": "Test Offer",
                "priority_score": 50.0,
                "key_factors": [{"feature": "age", "value": 45, "shap_value": 0.5, "direction": "increases"}]
            },
            client=llm_client,
            model=llm_model
        )
        if test_response.startswith("[Error"):
            print(f"   ⚠️ Test failed: {test_response}")
            print("   → Check your endpoint name and permissions")
        else:
            print(f"   ✓ LLM connection successful!")
            print(f"   Sample response: {test_response[:100]}...")
    except Exception as e:
        print(f"   ⚠️ Connection test failed: {e}")
else:
    print("⚠️ LLM client not available - skipping reasoning generation")

# COMMAND ----------

# Generate reasoning for sample members
if llm_client:
    reasoned_explanations = generate_all_offer_reasonings(
        llm_export=llm_export,
        client=llm_client,
        model=llm_model,
        max_members=5  # Limit for demo; remove for full run
    )
else:
    reasoned_explanations = None
    print("⚠️ Skipping reasoning generation - no LLM client available")

# COMMAND ----------

# MAGIC %md
# MAGIC ### View Generated Reasoning

# COMMAND ----------

if reasoned_explanations:
    print("\n" + "=" * 80)
    print("🤖 LLM-GENERATED OFFER REASONING")
    print("=" * 80)
    
    for member in reasoned_explanations[:3]:  # Show first 3 members
        print(f"\n👤 MEMBER: {member['member_id']}")
        print(f"   Age: {member['profile']['age']} | Risk Score: {member['profile']['risk_score']:.1f}")
        print("-" * 80)
        
        for rec in member['recommendations']:
            print(f"\n   📋 [{rec['rank']}] {rec['offer_name']}")
            print(f"   Score: {rec['priority_score']:.1f}/100")
            print(f"\n   💬 Why this offer?")
            # Word wrap the reasoning
            reasoning = rec['reasoning']
            words = reasoning.split()
            line = "      "
            for word in words:
                if len(line) + len(word) + 1 > 85:
                    print(line)
                    line = "      " + word
                else:
                    line += " " + word if line.strip() else word
            if line.strip():
                print(line)
        
        print("\n" + "=" * 80)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Export Recommendations with Reasoning

# COMMAND ----------

if reasoned_explanations:
    # Create a flat DataFrame for easy export
    reasoning_records = []
    
    for member in reasoned_explanations:
        for rec in member['recommendations']:
            reasoning_records.append({
                'member_id': member['member_id'],
                'age': member['profile']['age'],
                'risk_score': member['profile']['risk_score'],
                'chronic_conditions': member['profile']['chronic_condition_count'],
                'rank': rec['rank'],
                'offer_id': rec['offer_id'],
                'offer_name': rec['offer_name'],
                'priority_score': rec['priority_score'],
                'reasoning': rec['reasoning']
            })
    
    reasoning_df = pd.DataFrame(reasoning_records)
    
    print("\n📊 Recommendations with LLM Reasoning:")
    print(reasoning_df.head(10).to_string(index=False))
    
    # Save to JSON
    with open("/tmp/recommendations_with_reasoning.json", "w") as f:
        json.dump(reasoned_explanations, f, indent=2)
    
    # Save to CSV
    reasoning_df.to_csv("/tmp/recommendations_with_reasoning.csv", index=False)
    
    print(f"\n✅ Saved reasoning to:")
    print(f"   - /tmp/recommendations_with_reasoning.json")
    print(f"   - /tmp/recommendations_with_reasoning.csv")

