# Databricks notebook source
# MAGIC %md
# MAGIC # 01 - Data Exploration & Generation
# MAGIC 
# MAGIC This notebook generates synthetic healthcare data and performs exploratory data analysis
# MAGIC for the offer prioritization demo.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Setup

# COMMAND ----------

# Install required packages (if not already installed)
# %pip install lightgbm shap plotly

# COMMAND ----------

import sys
import os

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(".")))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# COMMAND ----------

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# Set display options
pd.set_option('display.max_columns', 50)
pd.set_option('display.max_rows', 100)

# Plot settings
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

# Import project modules
# In Databricks, use: %run ../config/config
# For local development:
try:
    from config.config import databricks_config, data_config, print_config
except ImportError:
    # Define inline for notebook execution
    class DatabricksConfig:
        catalog_name = "healthcare_demo"
        schema_name = "offer_prioritization"
        experiment_name = "/Shared/healthcare_offer_prioritization"
        model_name = "healthcare_offer_prioritizer"
    
    class DataConfig:
        num_members = 50000
        members_table = "members"
        claims_table = "claims"
        benefits_table = "benefits_utilization"
        engagement_table = "engagement_history"
    
    databricks_config = DatabricksConfig()
    data_config = DataConfig()

# COMMAND ----------

# Print configuration
print("=" * 60)
print("Healthcare Offer Prioritization Demo")
print("=" * 60)
print(f"\n📦 Catalog: {databricks_config.catalog_name}")
print(f"📁 Schema: {databricks_config.schema_name}")
print(f"👥 Members to generate: {data_config.num_members:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Generate Synthetic Data

# COMMAND ----------

# Import data generator
try:
    from data.generate_synthetic_data import generate_all_data, SyntheticDataGenerator
except ImportError:
    # Use %run in Databricks: %run ../data/generate_synthetic_data
    exec(open("../data/generate_synthetic_data.py").read())

# COMMAND ----------

# Generate data
members_df, claims_df, benefits_df, engagement_df = generate_all_data(
    n_members=data_config.num_members,
    claims_lookback=730,
    engagement_lookback=365,
    seed=42
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Overview

# COMMAND ----------

# MAGIC %md
# MAGIC ### Members Data

# COMMAND ----------

print(f"Members: {len(members_df):,} records")
print(f"Columns: {list(members_df.columns)}")
members_df.head(10)

# COMMAND ----------

# Member demographics summary
print("\n📊 Member Demographics Summary")
print("=" * 50)
print(f"\nAge Distribution:")
print(members_df['age'].describe())

print(f"\nGender Distribution:")
print(members_df['gender'].value_counts(normalize=True))

print(f"\nRegion Distribution:")
print(members_df['region'].value_counts(normalize=True))

print(f"\nPlan Type Distribution:")
print(members_df['plan_type'].value_counts(normalize=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Claims Data

# COMMAND ----------

print(f"Claims: {len(claims_df):,} records")
print(f"Columns: {list(claims_df.columns)}")
claims_df.head(10)

# COMMAND ----------

# Claims summary
print("\n📋 Claims Summary")
print("=" * 50)
print(f"\nClaims per Member:")
claims_per_member = claims_df.groupby('member_id').size()
print(f"  Min: {claims_per_member.min()}")
print(f"  Mean: {claims_per_member.mean():.1f}")
print(f"  Median: {claims_per_member.median()}")
print(f"  Max: {claims_per_member.max()}")

print(f"\nClaim Amount Statistics:")
print(claims_df['claim_amount'].describe())

print(f"\nClaim Type Distribution:")
print(claims_df['claim_type'].value_counts(normalize=True))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Benefits Utilization Data

# COMMAND ----------

print(f"Benefits: {len(benefits_df):,} records")
print(f"Columns: {list(benefits_df.columns)}")
benefits_df.head(10)

# COMMAND ----------

# Benefits summary
print("\n💊 Benefits Utilization Summary")
print("=" * 50)

benefit_summary = benefits_df.groupby('benefit_type').agg({
    'utilization_rate': ['mean', 'std'],
    'used_amount': 'sum',
    'remaining_balance': 'sum'
}).round(3)

print(benefit_summary)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Engagement Data

# COMMAND ----------

print(f"Engagement: {len(engagement_df):,} records")
print(f"Columns: {list(engagement_df.columns)}")
engagement_df.head(10)

# COMMAND ----------

# Engagement summary
print("\n📱 Engagement Summary")
print("=" * 50)

print(f"\nChannel Distribution:")
print(engagement_df['channel'].value_counts(normalize=True))

print(f"\nEngagement Type Distribution:")
print(engagement_df['engagement_type'].value_counts(normalize=True))

print(f"\nOverall Response Rate: {engagement_df['response_flag'].mean():.2%}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Visualizations

# COMMAND ----------

# Create visualization figure
fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 1. Age distribution
ax1 = axes[0, 0]
members_df['age'].hist(bins=30, ax=ax1, color='steelblue', edgecolor='white')
ax1.set_title('Age Distribution', fontsize=12, fontweight='bold')
ax1.set_xlabel('Age')
ax1.set_ylabel('Count')

# 2. Plan type distribution
ax2 = axes[0, 1]
members_df['plan_type'].value_counts().plot(kind='bar', ax=ax2, color='coral')
ax2.set_title('Plan Type Distribution', fontsize=12, fontweight='bold')
ax2.set_xlabel('Plan Type')
ax2.tick_params(axis='x', rotation=45)

# 3. Claim amount distribution
ax3 = axes[0, 2]
claims_df['claim_amount'].clip(upper=5000).hist(bins=50, ax=ax3, color='seagreen')
ax3.set_title('Claim Amount Distribution (capped at $5K)', fontsize=12, fontweight='bold')
ax3.set_xlabel('Claim Amount ($)')

# 4. Claims by type
ax4 = axes[1, 0]
claims_df['claim_type'].value_counts().plot(kind='barh', ax=ax4, color='mediumpurple')
ax4.set_title('Claims by Type', fontsize=12, fontweight='bold')
ax4.set_xlabel('Count')

# 5. Benefit utilization rates
ax5 = axes[1, 1]
util_by_type = benefits_df.groupby('benefit_type')['utilization_rate'].mean().sort_values()
util_by_type.plot(kind='barh', ax=ax5, color='goldenrod')
ax5.set_title('Avg Utilization Rate by Benefit', fontsize=12, fontweight='bold')
ax5.set_xlabel('Utilization Rate')

# 6. Engagement by channel
ax6 = axes[1, 2]
engagement_df['channel'].value_counts().plot(kind='pie', ax=ax6, autopct='%1.1f%%')
ax6.set_title('Engagement by Channel', fontsize=12, fontweight='bold')
ax6.set_ylabel('')

plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Correlation Analysis

# COMMAND ----------

# Analyze relationships between member characteristics and claims
member_claims_summary = claims_df.groupby('member_id').agg({
    'claim_id': 'count',
    'claim_amount': 'sum'
}).reset_index()
member_claims_summary.columns = ['member_id', 'claim_count', 'total_claim_amount']

analysis_df = members_df.merge(member_claims_summary, on='member_id', how='left')
analysis_df['claim_count'] = analysis_df['claim_count'].fillna(0)
analysis_df['total_claim_amount'] = analysis_df['total_claim_amount'].fillna(0)

# COMMAND ----------

# Correlation matrix
numeric_cols = ['age', 'tenure_months', 'family_size', 'risk_score', 
                'chronic_condition_count', 'claim_count', 'total_claim_amount']

corr_matrix = analysis_df[numeric_cols].corr()

plt.figure(figsize=(10, 8))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
            fmt='.2f', linewidths=0.5)
plt.title('Correlation Matrix: Member Characteristics vs Claims', 
          fontsize=14, fontweight='bold')
plt.tight_layout()
plt.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Save Data to Delta Tables (Databricks)

# COMMAND ----------

# MAGIC %md
# MAGIC Uncomment and run the following cells to save data to Delta tables in Databricks:

# COMMAND ----------

# # Create catalog and schema if needed
# spark.sql(f"CREATE CATALOG IF NOT EXISTS {databricks_config.catalog_name}")
# spark.sql(f"CREATE SCHEMA IF NOT EXISTS {databricks_config.catalog_name}.{databricks_config.schema_name}")

# COMMAND ----------

# # Convert pandas DataFrames to Spark and save as Delta tables
# 
# # Members table
# members_spark = spark.createDataFrame(members_df)
# members_spark.write.mode("overwrite").saveAsTable(
#     f"{databricks_config.catalog_name}.{databricks_config.schema_name}.{data_config.members_table}"
# )
# print(f"✓ Saved members table")
# 
# # Claims table
# claims_spark = spark.createDataFrame(claims_df)
# claims_spark.write.mode("overwrite").saveAsTable(
#     f"{databricks_config.catalog_name}.{databricks_config.schema_name}.{data_config.claims_table}"
# )
# print(f"✓ Saved claims table")
# 
# # Benefits table
# benefits_spark = spark.createDataFrame(benefits_df)
# benefits_spark.write.mode("overwrite").saveAsTable(
#     f"{databricks_config.catalog_name}.{databricks_config.schema_name}.{data_config.benefits_table}"
# )
# print(f"✓ Saved benefits table")
# 
# # Engagement table
# engagement_spark = spark.createDataFrame(engagement_df)
# engagement_spark.write.mode("overwrite").saveAsTable(
#     f"{databricks_config.catalog_name}.{databricks_config.schema_name}.{data_config.engagement_table}"
# )
# print(f"✓ Saved engagement table")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Summary Statistics for Next Steps

# COMMAND ----------

print("\n" + "=" * 60)
print("📊 DATA SUMMARY")
print("=" * 60)
print(f"\n👥 Members: {len(members_df):,}")
print(f"📋 Claims: {len(claims_df):,}")
print(f"💊 Benefit Records: {len(benefits_df):,}")
print(f"📱 Engagement Records: {len(engagement_df):,}")
print(f"\n✅ Data is ready for feature engineering (Notebook 02)")
print("=" * 60)

