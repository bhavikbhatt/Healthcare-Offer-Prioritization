# Databricks notebook source
# MAGIC %md
# MAGIC # 02 - Feature Engineering
# MAGIC 
# MAGIC This notebook creates ML features from member, claims, benefits, and engagement data
# MAGIC for the offer prioritization model.

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

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Load Data
# MAGIC 
# MAGIC In Databricks, load from Delta tables. For local development, generate synthetic data.

# COMMAND ----------

# Option 1: Load from Delta tables (Databricks)
# members_df = spark.table("healthcare_demo.offer_prioritization.members").toPandas()
# claims_df = spark.table("healthcare_demo.offer_prioritization.claims").toPandas()
# benefits_df = spark.table("healthcare_demo.offer_prioritization.benefits_utilization").toPandas()
# engagement_df = spark.table("healthcare_demo.offer_prioritization.engagement_history").toPandas()

# Option 2: Generate synthetic data (for demo)
try:
    from data.generate_synthetic_data import generate_all_data
except ImportError:
    exec(open("../data/generate_synthetic_data.py").read())

members_df, claims_df, benefits_df, engagement_df = generate_all_data(
    n_members=50000,
    seed=42
)

# COMMAND ----------

print(f"📊 Data Loaded:")
print(f"   Members: {len(members_df):,}")
print(f"   Claims: {len(claims_df):,}")
print(f"   Benefits: {len(benefits_df):,}")
print(f"   Engagement: {len(engagement_df):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Engineering Pipeline

# COMMAND ----------

# Import feature engineering module
try:
    from features.feature_engineering import FeatureEngineer, create_member_features
except ImportError:
    exec(open("../features/feature_engineering.py").read())

# COMMAND ----------

# Create features
features_df, feature_engineer = create_member_features(
    members_df=members_df,
    claims_df=claims_df,
    benefits_df=benefits_df,
    engagement_df=engagement_df,
    reference_date=datetime.now()
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Summary

# COMMAND ----------

print(f"\n📊 Feature DataFrame Shape: {features_df.shape}")
print(f"\nFeatures created: {len(feature_engineer.feature_names)}")
features_df.head(10)

# COMMAND ----------

# Feature statistics
feature_stats = features_df.drop(columns=['member_id']).describe().T
feature_stats['missing_pct'] = (features_df.drop(columns=['member_id']).isnull().sum() / len(features_df) * 100)
feature_stats = feature_stats[['count', 'mean', 'std', 'min', 'max', 'missing_pct']]
feature_stats.round(3)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Groups Analysis

# COMMAND ----------

# Get feature groups
feature_groups = feature_engineer.get_feature_importance_groups()

print("\n📋 Feature Groups:")
print("=" * 50)
for group, features in feature_groups.items():
    print(f"\n{group.upper()} ({len(features)} features):")
    for f in features[:5]:
        print(f"  - {f}")
    if len(features) > 5:
        print(f"  ... and {len(features) - 5} more")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Distributions

# COMMAND ----------

# Select key features for visualization
key_features = [
    'age', 'tenure_months', 'risk_score', 'chronic_condition_count',
    'total_claims_count', 'total_claims_amount', 'avg_claim_amount',
    'avg_utilization_rate', 'total_engagements', 'avg_response_rate'
]

# Filter to available features
available_features = [f for f in key_features if f in features_df.columns]

# Create distribution plots
n_cols = 3
n_rows = (len(available_features) + n_cols - 1) // n_cols

fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4*n_rows))
axes = axes.flatten()

for i, feature in enumerate(available_features):
    ax = axes[i]
    data = features_df[feature].dropna()
    
    # Use log scale for highly skewed features
    if feature in ['total_claims_amount', 'total_claims_count']:
        data = data[data > 0]
        ax.hist(np.log1p(data), bins=50, color='steelblue', edgecolor='white')
        ax.set_xlabel(f'log({feature})')
    else:
        ax.hist(data, bins=50, color='steelblue', edgecolor='white')
        ax.set_xlabel(feature)
    
    ax.set_title(feature.replace('_', ' ').title(), fontweight='bold')
    ax.set_ylabel('Count')

# Hide unused subplots
for i in range(len(available_features), len(axes)):
    axes[i].set_visible(False)

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Feature Correlations

# COMMAND ----------

# Select numeric features for correlation
numeric_features = features_df.select_dtypes(include=[np.number]).columns.tolist()
numeric_features = [f for f in numeric_features if f != 'member_id']

# Limit to top features for readability
if len(numeric_features) > 20:
    # Select diverse set of features
    selected_features = [
        'age', 'tenure_months', 'risk_score', 'chronic_condition_count',
        'total_claims_count', 'total_claims_amount', 'claims_last_90d',
        'has_diabetes', 'has_cardiovascular', 'has_mental_health',
        'avg_utilization_rate', 'pharmacy_utilization_rate',
        'total_engagements', 'avg_response_rate', 'days_since_last_claim',
        'is_senior', 'high_risk_flag', 'is_complex_patient'
    ]
    selected_features = [f for f in selected_features if f in numeric_features]
else:
    selected_features = numeric_features

# Calculate correlation matrix
corr_matrix = features_df[selected_features].corr()

# Plot
plt.figure(figsize=(14, 12))
mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
sns.heatmap(corr_matrix, mask=mask, annot=True, cmap='RdBu_r', center=0,
            fmt='.2f', linewidths=0.5, annot_kws={'size': 8})
plt.title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## High-Value Member Segments

# COMMAND ----------

# Identify member segments based on features
features_df['segment'] = 'Standard'

# High-risk chronic
features_df.loc[
    (features_df['chronic_condition_count'] >= 2) & 
    (features_df['risk_score'] > 50), 
    'segment'
] = 'High-Risk Chronic'

# Active engagers
features_df.loc[
    (features_df['total_engagements'] > features_df['total_engagements'].quantile(0.75)) &
    (features_df['avg_response_rate'] > 0.3),
    'segment'
] = 'Active Engagers'

# High utilizers
features_df.loc[
    features_df['total_claims_count'] > features_df['total_claims_count'].quantile(0.9),
    'segment'
] = 'High Utilizers'

# New members
features_df.loc[features_df['is_new_member'] == 1, 'segment'] = 'New Members'

# Seniors
features_df.loc[features_df['is_senior'] == 1, 'segment'] = 'Seniors'

# COMMAND ----------

# Segment analysis
segment_summary = features_df.groupby('segment').agg({
    'member_id': 'count',
    'age': 'mean',
    'total_claims_count': 'mean',
    'total_claims_amount': 'mean',
    'avg_utilization_rate': 'mean',
    'total_engagements': 'mean',
    'risk_score': 'mean'
}).round(2)

segment_summary.columns = ['Count', 'Avg Age', 'Avg Claims', 'Avg Claim $', 
                           'Avg Utilization', 'Avg Engagements', 'Avg Risk Score']
segment_summary['Pct'] = (segment_summary['Count'] / len(features_df) * 100).round(1)

print("\n📊 Member Segments:")
print("=" * 80)
print(segment_summary.sort_values('Count', ascending=False))

# COMMAND ----------

# Visualize segments
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Segment distribution
ax1 = axes[0]
segment_counts = features_df['segment'].value_counts()
segment_counts.plot(kind='bar', ax=ax1, color='coral')
ax1.set_title('Member Segments Distribution', fontweight='bold')
ax1.set_xlabel('Segment')
ax1.set_ylabel('Count')
ax1.tick_params(axis='x', rotation=45)

# Average claims by segment
ax2 = axes[1]
segment_claims = features_df.groupby('segment')['total_claims_amount'].mean().sort_values()
segment_claims.plot(kind='barh', ax=ax2, color='steelblue')
ax2.set_title('Average Claims Amount by Segment', fontweight='bold')
ax2.set_xlabel('Average Claim Amount ($)')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Features to Delta Table

# COMMAND ----------

# # Save features to Delta table
# features_spark = spark.createDataFrame(features_df.drop(columns=['segment']))
# features_spark.write.mode("overwrite").saveAsTable(
#     "healthcare_demo.offer_prioritization.member_features"
# )
# print("✓ Saved member features table")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary

# COMMAND ----------

# Remove segment column for model training
features_for_model = features_df.drop(columns=['segment'])

print("\n" + "=" * 60)
print("📊 FEATURE ENGINEERING SUMMARY")
print("=" * 60)
print(f"\n👥 Members processed: {len(features_for_model):,}")
print(f"🔢 Features created: {len(feature_engineer.feature_names)}")
print(f"\n📋 Feature Groups:")
for group, features in feature_groups.items():
    print(f"   {group}: {len(features)} features")
print(f"\n✅ Features ready for model training (Notebook 03)")
print("=" * 60)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export for Next Notebook

# COMMAND ----------

# Save features DataFrame for use in next notebook
# In production, this would be read from Delta table
# features_for_model.to_parquet("/tmp/member_features.parquet", index=False)

