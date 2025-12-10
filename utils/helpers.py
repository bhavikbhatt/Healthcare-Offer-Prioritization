# Databricks notebook source
# MAGIC %md
# MAGIC # Utility Functions
# MAGIC Helper functions for the healthcare offer prioritization project

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple, Any
import json
import logging

# COMMAND ----------

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("offer_prioritization")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Data Validation Utilities

# COMMAND ----------

def validate_member_data(df: pd.DataFrame, required_columns: List[str] = None) -> Tuple[bool, List[str]]:
    """
    Validate member data DataFrame
    
    Args:
        df: DataFrame to validate
        required_columns: List of required column names
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    # Default required columns
    if required_columns is None:
        required_columns = ['member_id']
    
    # Check for required columns
    missing_cols = [col for col in required_columns if col not in df.columns]
    if missing_cols:
        errors.append(f"Missing required columns: {missing_cols}")
    
    # Check for duplicate member_ids
    if 'member_id' in df.columns:
        duplicates = df['member_id'].duplicated().sum()
        if duplicates > 0:
            errors.append(f"Found {duplicates} duplicate member_ids")
    
    # Check for empty DataFrame
    if len(df) == 0:
        errors.append("DataFrame is empty")
    
    # Check for null member_ids
    if 'member_id' in df.columns:
        null_ids = df['member_id'].isnull().sum()
        if null_ids > 0:
            errors.append(f"Found {null_ids} null member_ids")
    
    is_valid = len(errors) == 0
    
    if not is_valid:
        logger.warning(f"Data validation failed: {errors}")
    else:
        logger.info("Data validation passed")
    
    return is_valid, errors


def validate_features(df: pd.DataFrame, feature_names: List[str]) -> Tuple[bool, List[str]]:
    """
    Validate feature DataFrame
    
    Args:
        df: Feature DataFrame
        feature_names: Expected feature names
        
    Returns:
        Tuple of (is_valid, list of error messages)
    """
    errors = []
    
    # Check for missing features
    missing = [f for f in feature_names if f not in df.columns]
    if missing:
        errors.append(f"Missing features: {missing[:10]}..." if len(missing) > 10 else f"Missing features: {missing}")
    
    # Check for infinite values
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_counts = np.isinf(df[numeric_cols]).sum()
    inf_cols = inf_counts[inf_counts > 0].index.tolist()
    if inf_cols:
        errors.append(f"Infinite values in columns: {inf_cols}")
    
    # Check for high null rate
    null_rates = df.isnull().mean()
    high_null_cols = null_rates[null_rates > 0.5].index.tolist()
    if high_null_cols:
        errors.append(f"High null rate (>50%) in columns: {high_null_cols}")
    
    is_valid = len(errors) == 0
    return is_valid, errors

# COMMAND ----------

# MAGIC %md
# MAGIC ## Databricks Utilities

# COMMAND ----------

def get_table_path(catalog: str, schema: str, table: str) -> str:
    """Get full table path for Unity Catalog"""
    return f"{catalog}.{schema}.{table}"


def table_exists(spark, table_path: str) -> bool:
    """Check if a table exists in Unity Catalog"""
    try:
        spark.table(table_path)
        return True
    except:
        return False


def create_schema_if_not_exists(spark, catalog: str, schema: str):
    """Create catalog and schema if they don't exist"""
    spark.sql(f"CREATE CATALOG IF NOT EXISTS {catalog}")
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {catalog}.{schema}")
    logger.info(f"Ensured schema exists: {catalog}.{schema}")


def get_latest_model_version(client, model_name: str) -> int:
    """Get the latest version of a registered model"""
    from mlflow.tracking import MlflowClient
    
    if client is None:
        client = MlflowClient()
    
    versions = client.search_model_versions(f"name='{model_name}'")
    if not versions:
        return None
    
    return max(int(v.version) for v in versions)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Scoring Utilities

# COMMAND ----------

def normalize_scores(scores: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize scores to 0-100 range
    
    Args:
        scores: Raw scores array
        method: Normalization method ("minmax", "zscore", "percentile")
        
    Returns:
        Normalized scores
    """
    if method == "minmax":
        min_val = scores.min()
        max_val = scores.max()
        if max_val == min_val:
            return np.full_like(scores, 50.0)
        normalized = (scores - min_val) / (max_val - min_val) * 100
    
    elif method == "zscore":
        mean = scores.mean()
        std = scores.std()
        if std == 0:
            return np.full_like(scores, 50.0)
        z_scores = (scores - mean) / std
        # Convert z-scores to 0-100 range (assuming ~99.7% within 3 std)
        normalized = (z_scores + 3) / 6 * 100
        normalized = np.clip(normalized, 0, 100)
    
    elif method == "percentile":
        from scipy import stats
        normalized = stats.rankdata(scores, method='average') / len(scores) * 100
    
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    return normalized


def apply_business_rules(
    recommendations: pd.DataFrame,
    member_features: pd.DataFrame,
    rules: Dict[str, Any] = None
) -> pd.DataFrame:
    """
    Apply business rules to filter/adjust recommendations
    
    Args:
        recommendations: Raw recommendations DataFrame
        member_features: Member features for rule application
        rules: Dictionary of business rules
        
    Returns:
        Filtered/adjusted recommendations
    """
    if rules is None:
        rules = {}
    
    result = recommendations.copy()
    
    # Example rule: Exclude certain offers for members who recently received them
    if 'cooldown_days' in rules:
        # Implementation would check engagement history
        pass
    
    # Example rule: Boost scores for high-value members
    if 'high_value_boost' in rules:
        # Implementation would identify high-value members and boost
        pass
    
    # Example rule: Cap number of offers per category
    if 'max_per_category' in rules:
        max_cat = rules['max_per_category']
        result['cat_rank'] = result.groupby(['member_id', 'category']).cumcount() + 1
        result = result[result['cat_rank'] <= max_cat]
        result = result.drop(columns=['cat_rank'])
    
    return result

# COMMAND ----------

# MAGIC %md
# MAGIC ## Reporting Utilities

# COMMAND ----------

def generate_recommendation_summary(
    recommendations: pd.DataFrame,
    member_features: pd.DataFrame = None
) -> Dict[str, Any]:
    """
    Generate summary statistics for recommendations
    
    Args:
        recommendations: Recommendations DataFrame
        member_features: Optional member features for segmentation
        
    Returns:
        Dictionary with summary statistics
    """
    summary = {
        "generated_at": datetime.now().isoformat(),
        "total_recommendations": len(recommendations),
        "unique_members": recommendations['member_id'].nunique(),
        "unique_offers": recommendations['offer_id'].nunique(),
        "avg_recommendations_per_member": len(recommendations) / recommendations['member_id'].nunique(),
        "score_statistics": {
            "mean": float(recommendations['priority_score'].mean()),
            "std": float(recommendations['priority_score'].std()),
            "min": float(recommendations['priority_score'].min()),
            "max": float(recommendations['priority_score'].max()),
            "median": float(recommendations['priority_score'].median())
        },
        "top_offers": recommendations['offer_name'].value_counts().head(5).to_dict(),
        "category_distribution": recommendations['category'].value_counts().to_dict()
    }
    
    # Add segment analysis if member features provided
    if member_features is not None and 'member_id' in member_features.columns:
        merged = recommendations.merge(
            member_features[['member_id', 'age', 'risk_score']], 
            on='member_id',
            how='left'
        )
        
        # Age group analysis
        merged['age_group'] = pd.cut(
            merged['age'], 
            bins=[0, 35, 50, 65, 100],
            labels=['<35', '35-50', '50-65', '65+']
        )
        summary['recommendations_by_age_group'] = merged.groupby('age_group').size().to_dict()
    
    return summary


def format_recommendation_report(summary: Dict[str, Any]) -> str:
    """
    Format recommendation summary as readable report
    
    Args:
        summary: Summary dictionary from generate_recommendation_summary
        
    Returns:
        Formatted report string
    """
    report = []
    report.append("=" * 60)
    report.append("OFFER RECOMMENDATION REPORT")
    report.append("=" * 60)
    report.append(f"\nGenerated: {summary['generated_at']}")
    report.append(f"\n📊 Overview:")
    report.append(f"   Total Recommendations: {summary['total_recommendations']:,}")
    report.append(f"   Unique Members: {summary['unique_members']:,}")
    report.append(f"   Avg per Member: {summary['avg_recommendations_per_member']:.1f}")
    
    report.append(f"\n📈 Score Statistics:")
    stats = summary['score_statistics']
    report.append(f"   Mean: {stats['mean']:.1f}")
    report.append(f"   Median: {stats['median']:.1f}")
    report.append(f"   Range: {stats['min']:.1f} - {stats['max']:.1f}")
    
    report.append(f"\n🏆 Top Offers:")
    for offer, count in summary['top_offers'].items():
        report.append(f"   • {offer}: {count:,}")
    
    report.append(f"\n📋 By Category:")
    for category, count in summary['category_distribution'].items():
        report.append(f"   • {category}: {count:,}")
    
    report.append("\n" + "=" * 60)
    
    return "\n".join(report)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Export Utilities

# COMMAND ----------

def export_to_json(
    recommendations: pd.DataFrame,
    output_path: str,
    include_metadata: bool = True
) -> str:
    """
    Export recommendations to JSON format
    
    Args:
        recommendations: Recommendations DataFrame
        output_path: Output file path
        include_metadata: Whether to include metadata
        
    Returns:
        Output file path
    """
    output = {
        "recommendations": recommendations.to_dict('records')
    }
    
    if include_metadata:
        output["metadata"] = {
            "generated_at": datetime.now().isoformat(),
            "total_records": len(recommendations),
            "unique_members": int(recommendations['member_id'].nunique())
        }
    
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2, default=str)
    
    logger.info(f"Exported {len(recommendations)} recommendations to {output_path}")
    return output_path


def prepare_for_marketing_platform(
    recommendations: pd.DataFrame,
    platform: str = "generic"
) -> pd.DataFrame:
    """
    Prepare recommendations for marketing platform integration
    
    Args:
        recommendations: Recommendations DataFrame
        platform: Target platform (generic, salesforce, marketo, etc.)
        
    Returns:
        Platform-ready DataFrame
    """
    export_df = recommendations.copy()
    
    # Add common fields
    export_df['created_date'] = datetime.now().strftime('%Y-%m-%d')
    export_df['campaign_type'] = 'healthcare_offers'
    
    if platform == "salesforce":
        # Rename columns for Salesforce
        export_df = export_df.rename(columns={
            'member_id': 'ContactId',
            'offer_id': 'CampaignMemberId',
            'priority_score': 'Score__c',
            'offer_name': 'OfferName__c'
        })
    
    elif platform == "marketo":
        # Rename columns for Marketo
        export_df = export_df.rename(columns={
            'member_id': 'leadId',
            'offer_id': 'programId',
            'priority_score': 'score',
            'offer_name': 'programName'
        })
    
    return export_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Performance Monitoring

# COMMAND ----------

class ModelMonitor:
    """
    Monitor model performance and data drift
    """
    
    def __init__(self, baseline_stats: Dict = None):
        self.baseline_stats = baseline_stats or {}
        self.current_stats = {}
        
    def compute_feature_stats(self, df: pd.DataFrame) -> Dict:
        """Compute statistics for features"""
        stats = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        
        for col in numeric_cols:
            stats[col] = {
                'mean': float(df[col].mean()),
                'std': float(df[col].std()),
                'min': float(df[col].min()),
                'max': float(df[col].max()),
                'null_rate': float(df[col].isnull().mean())
            }
        
        return stats
    
    def set_baseline(self, df: pd.DataFrame):
        """Set baseline statistics"""
        self.baseline_stats = self.compute_feature_stats(df)
        logger.info("Baseline statistics set")
    
    def check_drift(self, df: pd.DataFrame, threshold: float = 0.1) -> Dict[str, bool]:
        """
        Check for data drift against baseline
        
        Args:
            df: Current data DataFrame
            threshold: Drift threshold (relative change)
            
        Returns:
            Dictionary of feature: has_drift
        """
        if not self.baseline_stats:
            logger.warning("No baseline set - cannot check drift")
            return {}
        
        self.current_stats = self.compute_feature_stats(df)
        drift_results = {}
        
        for feature, baseline in self.baseline_stats.items():
            if feature not in self.current_stats:
                continue
            
            current = self.current_stats[feature]
            
            # Check mean shift
            if baseline['mean'] != 0:
                mean_change = abs(current['mean'] - baseline['mean']) / abs(baseline['mean'])
            else:
                mean_change = abs(current['mean'] - baseline['mean'])
            
            drift_results[feature] = mean_change > threshold
        
        drifted_features = [f for f, d in drift_results.items() if d]
        if drifted_features:
            logger.warning(f"Data drift detected in features: {drifted_features}")
        
        return drift_results

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample Usage

# COMMAND ----------

# # Validation example
# is_valid, errors = validate_member_data(members_df)
# print(f"Data valid: {is_valid}")
# if errors:
#     print(f"Errors: {errors}")

# # Summary report example
# summary = generate_recommendation_summary(recommendations, features_df)
# report = format_recommendation_report(summary)
# print(report)

# # Export example
# export_to_json(recommendations.head(100), "/tmp/recommendations.json")

