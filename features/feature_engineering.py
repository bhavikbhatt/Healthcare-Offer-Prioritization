# Databricks notebook source
# MAGIC %md
# MAGIC # Feature Engineering Pipeline
# MAGIC Transform raw member data into ML-ready features for offer prioritization

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Optional
from sklearn.preprocessing import LabelEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

# COMMAND ----------

class FeatureEngineer:
    """
    Feature engineering pipeline for healthcare offer prioritization.
    Transforms raw member, claims, benefits, and engagement data into ML features.
    """
    
    def __init__(self, reference_date: Optional[datetime] = None):
        """
        Initialize feature engineer
        
        Args:
            reference_date: Date to use for time-based calculations (default: now)
        """
        self.reference_date = reference_date or datetime.now()
        self.label_encoders: Dict[str, LabelEncoder] = {}
        self.scalers: Dict[str, StandardScaler] = {}
        self.feature_names: List[str] = []
        
    def create_features(
        self,
        members_df: pd.DataFrame,
        claims_df: pd.DataFrame,
        benefits_df: pd.DataFrame,
        engagement_df: pd.DataFrame
    ) -> pd.DataFrame:
        """
        Create all features from raw data
        
        Args:
            members_df: Member demographics data
            claims_df: Claims history data
            benefits_df: Benefits utilization data
            engagement_df: Engagement history data
            
        Returns:
            DataFrame with all features joined on member_id
        """
        print("🔧 Starting Feature Engineering Pipeline")
        print("=" * 60)
        
        # Create feature groups
        print("\n📊 Creating demographic features...")
        demo_features = self._create_demographic_features(members_df)
        print(f"   ✓ {len([c for c in demo_features.columns if c != 'member_id'])} features")
        
        print("\n🏥 Creating claims features...")
        claims_features = self._create_claims_features(claims_df)
        print(f"   ✓ {len([c for c in claims_features.columns if c != 'member_id'])} features")
        
        print("\n🩺 Creating diagnosis features...")
        diagnosis_features = self._create_diagnosis_features(claims_df)
        print(f"   ✓ {len([c for c in diagnosis_features.columns if c != 'member_id'])} features")
        
        print("\n💊 Creating benefits features...")
        benefits_features = self._create_benefits_features(benefits_df)
        print(f"   ✓ {len([c for c in benefits_features.columns if c != 'member_id'])} features")
        
        print("\n📱 Creating engagement features...")
        engagement_features = self._create_engagement_features(engagement_df)
        print(f"   ✓ {len([c for c in engagement_features.columns if c != 'member_id'])} features")
        
        print("\n📅 Creating temporal features...")
        temporal_features = self._create_temporal_features(members_df)
        print(f"   ✓ {len([c for c in temporal_features.columns if c != 'member_id'])} features")
        
        # Merge all features
        print("\n🔗 Merging feature groups...")
        features_df = demo_features.copy()
        
        for df in [claims_features, diagnosis_features, benefits_features, 
                   engagement_features, temporal_features]:
            features_df = features_df.merge(df, on="member_id", how="left")
        
        # Fill missing values
        features_df = self._handle_missing_values(features_df)
        
        # Store feature names (excluding member_id)
        self.feature_names = [c for c in features_df.columns if c != "member_id"]
        
        print(f"\n✅ Feature engineering complete!")
        print(f"   Total features: {len(self.feature_names)}")
        print(f"   Total members: {len(features_df):,}")
        print("=" * 60)
        
        return features_df
    
    def _create_demographic_features(self, members_df: pd.DataFrame) -> pd.DataFrame:
        """Create demographic features"""
        df = members_df[["member_id"]].copy()
        
        # Numeric features
        df["age"] = members_df["age"]
        df["tenure_months"] = members_df["tenure_months"]
        df["family_size"] = members_df["family_size"]
        df["risk_score"] = members_df["risk_score"]
        df["chronic_condition_count"] = members_df["chronic_condition_count"]
        
        # Age group buckets
        df["age_group"] = pd.cut(
            members_df["age"],
            bins=[0, 26, 35, 50, 65, 100],
            labels=["18-25", "26-35", "36-50", "51-65", "65+"]
        ).astype(str)
        
        # Tenure buckets
        df["tenure_group"] = pd.cut(
            members_df["tenure_months"],
            bins=[0, 12, 24, 60, 120, 300],
            labels=["<1yr", "1-2yr", "2-5yr", "5-10yr", "10yr+"]
        ).astype(str)
        
        # Encode categorical features from members_df
        categorical_cols_from_members = ["gender", "region", "plan_type", "income_bracket"]
        for col in categorical_cols_from_members:
            if col in members_df.columns:
                df[f"{col}_encoded"] = self._encode_categorical(members_df[col], col)
        
        # Encode categorical features created locally in df
        categorical_cols_from_df = ["age_group", "tenure_group"]
        for col in categorical_cols_from_df:
            if col in df.columns:
                df[f"{col}_encoded"] = self._encode_categorical(df[col], col)
        
        # Drop the raw string columns (keep only encoded versions)
        df = df.drop(columns=["age_group", "tenure_group"])
        
        # Derived features
        df["is_senior"] = (members_df["age"] >= 65).astype(int)
        df["is_new_member"] = (members_df["tenure_months"] <= 12).astype(int)
        df["has_chronic_condition"] = (members_df["chronic_condition_count"] > 0).astype(int)
        df["high_risk_flag"] = (members_df["risk_score"] > 70).astype(int)
        
        return df
    
    def _create_claims_features(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """Create claims-based features"""
        
        # Aggregate claims by member
        claims_agg = claims_df.groupby("member_id").agg({
            "claim_id": "count",
            "claim_amount": ["sum", "mean", "std", "max"],
            "paid_amount": ["sum", "mean"],
            "member_responsibility": ["sum", "mean"],
            "in_network": "mean"
        }).reset_index()
        
        # Flatten column names
        claims_agg.columns = [
            "member_id", "total_claims_count", 
            "total_claims_amount", "avg_claim_amount", "std_claim_amount", "max_claim_amount",
            "total_paid_amount", "avg_paid_amount",
            "total_member_cost", "avg_member_cost",
            "in_network_rate"
        ]
        
        # Time-based claim counts
        now = self.reference_date
        for days, suffix in [(30, "30d"), (90, "90d"), (180, "180d"), (365, "365d")]:
            cutoff = now - timedelta(days=days)
            recent = claims_df[claims_df["claim_date"] >= cutoff]
            recent_counts = recent.groupby("member_id")["claim_id"].count().reset_index()
            recent_counts.columns = ["member_id", f"claims_last_{suffix}"]
            claims_agg = claims_agg.merge(recent_counts, on="member_id", how="left")
        
        # Claims by type
        type_pivot = claims_df.pivot_table(
            index="member_id",
            columns="claim_type",
            values="claim_id",
            aggfunc="count",
            fill_value=0
        ).reset_index()
        
        type_pivot.columns = ["member_id"] + [f"{c}_visit_count" for c in type_pivot.columns[1:]]
        claims_agg = claims_agg.merge(type_pivot, on="member_id", how="left")
        
        # Days since last claim
        last_claim = claims_df.groupby("member_id")["claim_date"].max().reset_index()
        last_claim.columns = ["member_id", "last_claim_date"]
        last_claim["days_since_last_claim"] = (now - last_claim["last_claim_date"]).dt.days
        claims_agg = claims_agg.merge(
            last_claim[["member_id", "days_since_last_claim"]], 
            on="member_id", 
            how="left"
        )
        
        # Claim frequency trend (recent vs older)
        if "claims_last_180d" in claims_agg.columns and "claims_last_365d" in claims_agg.columns:
            claims_agg["claim_frequency_trend"] = (
                claims_agg["claims_last_180d"].fillna(0) / 
                (claims_agg["claims_last_365d"].fillna(1) + 1)
            )
        
        # Cost trend
        claims_agg["cost_per_claim"] = (
            claims_agg["total_claims_amount"] / 
            (claims_agg["total_claims_count"] + 1)
        )
        
        return claims_agg
    
    def _create_diagnosis_features(self, claims_df: pd.DataFrame) -> pd.DataFrame:
        """Create diagnosis-based features"""
        
        # Diagnosis category flags
        diagnosis_pivot = claims_df.pivot_table(
            index="member_id",
            columns="diagnosis_category",
            values="claim_id",
            aggfunc="count",
            fill_value=0
        ).reset_index()
        
        # Create binary flags for key conditions
        df = diagnosis_pivot[["member_id"]].copy()
        
        condition_mapping = {
            "has_diabetes": "diabetes",
            "has_cardiovascular": "cardiovascular",
            "has_respiratory": "respiratory",
            "has_mental_health": "mental_health",
            "has_musculoskeletal": "musculoskeletal",
            "has_gastrointestinal": "gastrointestinal",
            "has_neurological": "neurological",
            "has_oncology": "oncology"
        }
        
        for flag_name, condition in condition_mapping.items():
            if condition in diagnosis_pivot.columns:
                df[flag_name] = (diagnosis_pivot[condition] > 0).astype(int)
            else:
                df[flag_name] = 0
        
        # Count unique diagnosis categories
        df["unique_diagnosis_count"] = (diagnosis_pivot.iloc[:, 1:] > 0).sum(axis=1)
        
        # High utilizer flag (multiple conditions)
        df["is_complex_patient"] = (df["unique_diagnosis_count"] >= 3).astype(int)
        
        return df
    
    def _create_benefits_features(self, benefits_df: pd.DataFrame) -> pd.DataFrame:
        """Create benefits utilization features"""
        
        # Pivot utilization rates by benefit type
        util_pivot = benefits_df.pivot_table(
            index="member_id",
            columns="benefit_type",
            values="utilization_rate",
            aggfunc="mean"
        ).reset_index()
        
        util_pivot.columns = ["member_id"] + [
            f"{c}_utilization_rate" for c in util_pivot.columns[1:]
        ]
        
        # Aggregate metrics
        benefits_agg = benefits_df.groupby("member_id").agg({
            "used_amount": "sum",
            "remaining_balance": "sum",
            "annual_max": "sum",
            "utilization_rate": ["mean", "max", "std"],
            "claims_count": "sum"
        }).reset_index()
        
        benefits_agg.columns = [
            "member_id", "total_benefits_used", "total_remaining_balance",
            "total_annual_max", "avg_utilization_rate", "max_utilization_rate",
            "std_utilization_rate", "total_benefit_claims"
        ]
        
        # Calculate remaining percentages
        benefits_agg["remaining_deductible_pct"] = (
            benefits_agg["total_remaining_balance"] / 
            (benefits_agg["total_annual_max"] + 1)
        )
        
        # Merge pivot with aggregates
        benefits_features = util_pivot.merge(benefits_agg, on="member_id", how="left")
        
        # Days since last benefit use
        last_use = benefits_df.dropna(subset=["last_used_date"]).groupby("member_id")["last_used_date"].max().reset_index()
        last_use.columns = ["member_id", "last_benefit_use_date"]
        last_use["days_since_last_benefit_use"] = (
            self.reference_date - last_use["last_benefit_use_date"]
        ).dt.days
        
        benefits_features = benefits_features.merge(
            last_use[["member_id", "days_since_last_benefit_use"]],
            on="member_id",
            how="left"
        )
        
        return benefits_features
    
    def _create_engagement_features(self, engagement_df: pd.DataFrame) -> pd.DataFrame:
        """Create engagement features"""
        
        # Aggregate engagement metrics
        eng_agg = engagement_df.groupby("member_id").agg({
            "engagement_id": "count",
            "response_flag": "mean",
            "session_duration_sec": "mean"
        }).reset_index()
        
        eng_agg.columns = [
            "member_id", "total_engagements", 
            "avg_response_rate", "avg_session_duration"
        ]
        
        # Channel breakdown
        channel_pivot = engagement_df.pivot_table(
            index="member_id",
            columns="channel",
            values="engagement_id",
            aggfunc="count",
            fill_value=0
        ).reset_index()
        
        total_eng = channel_pivot.iloc[:, 1:].sum(axis=1)
        for col in channel_pivot.columns[1:]:
            channel_pivot[f"{col}_engagement_rate"] = channel_pivot[col] / (total_eng + 1)
        
        # Keep only rate columns
        rate_cols = ["member_id"] + [c for c in channel_pivot.columns if "rate" in c]
        channel_rates = channel_pivot[rate_cols]
        
        # Engagement type breakdown
        type_pivot = engagement_df.pivot_table(
            index="member_id",
            columns="engagement_type",
            values="engagement_id",
            aggfunc="count",
            fill_value=0
        ).reset_index()
        
        type_pivot.columns = ["member_id"] + [f"{c}_count" for c in type_pivot.columns[1:]]
        
        # Preferred channel (most used)
        channel_only = channel_pivot[["member_id"] + [c for c in channel_pivot.columns if c not in ["member_id"] and "rate" not in c]]
        channel_only["preferred_channel"] = channel_only.iloc[:, 1:].idxmax(axis=1)
        channel_only["preferred_channel_encoded"] = self._encode_categorical(
            channel_only["preferred_channel"], "preferred_channel"
        )
        
        # Days since last engagement
        last_eng = engagement_df.groupby("member_id")["engagement_date"].max().reset_index()
        last_eng.columns = ["member_id", "last_engagement_date"]
        last_eng["days_since_last_engagement"] = (
            self.reference_date - last_eng["last_engagement_date"]
        ).dt.days
        
        # Recent engagement trend
        now = self.reference_date
        for days, suffix in [(30, "30d"), (90, "90d")]:
            cutoff = now - timedelta(days=days)
            recent = engagement_df[engagement_df["engagement_date"] >= cutoff]
            recent_counts = recent.groupby("member_id")["engagement_id"].count().reset_index()
            recent_counts.columns = ["member_id", f"engagements_last_{suffix}"]
            eng_agg = eng_agg.merge(recent_counts, on="member_id", how="left")
        
        # Merge all engagement features
        engagement_features = eng_agg.merge(channel_rates, on="member_id", how="left")
        engagement_features = engagement_features.merge(type_pivot, on="member_id", how="left")
        engagement_features = engagement_features.merge(
            channel_only[["member_id", "preferred_channel_encoded"]], 
            on="member_id", 
            how="left"
        )
        engagement_features = engagement_features.merge(
            last_eng[["member_id", "days_since_last_engagement"]], 
            on="member_id", 
            how="left"
        )
        
        # Engagement trend
        if "engagements_last_30d" in engagement_features.columns:
            engagement_features["engagement_trend"] = (
                engagement_features["engagements_last_30d"].fillna(0) / 
                (engagement_features["total_engagements"].fillna(1) + 1) * 12
            )
        
        return engagement_features
    
    def _create_temporal_features(self, members_df: pd.DataFrame) -> pd.DataFrame:
        """Create temporal/seasonal features"""
        df = members_df[["member_id"]].copy()
        
        now = self.reference_date
        
        # Current time features
        df["current_month"] = now.month
        df["current_quarter"] = (now.month - 1) // 3 + 1
        df["is_q4"] = (df["current_quarter"] == 4).astype(int)  # Year-end engagement
        df["is_open_enrollment"] = int(now.month in [10, 11, 12])
        
        # Seasonal health features
        df["is_flu_season"] = int(now.month in [10, 11, 12, 1, 2, 3])
        df["is_allergy_season"] = int(now.month in [3, 4, 5, 9, 10])
        
        # Days until year end (for benefits)
        year_end = datetime(now.year, 12, 31)
        df["days_until_year_end"] = (year_end - now).days
        df["benefits_urgency"] = 1 - (df["days_until_year_end"] / 365)
        
        return df
    
    def _encode_categorical(self, series: pd.Series, name: str) -> pd.Series:
        """Encode categorical variable"""
        if name not in self.label_encoders:
            self.label_encoders[name] = LabelEncoder()
            self.label_encoders[name].fit(series.astype(str).fillna("unknown"))
        
        return self.label_encoders[name].transform(series.astype(str).fillna("unknown"))
    
    def _handle_missing_values(self, df: pd.DataFrame) -> pd.DataFrame:
        """Handle missing values in features"""
        # Fill numeric columns with 0 for counts, median for rates
        for col in df.columns:
            if col == "member_id":
                continue
            
            if df[col].dtype in ["float64", "int64"]:
                if "count" in col or "total" in col:
                    df[col] = df[col].fillna(0)
                elif "rate" in col or "pct" in col:
                    df[col] = df[col].fillna(df[col].median())
                elif "days_since" in col:
                    df[col] = df[col].fillna(365)  # Assume no recent activity
                else:
                    df[col] = df[col].fillna(0)
        
        return df
    
    def get_feature_importance_groups(self) -> Dict[str, List[str]]:
        """Get features grouped by category for analysis"""
        return {
            "demographic": [f for f in self.feature_names if any(
                x in f for x in ["age", "gender", "region", "plan", "tenure", 
                                "income", "family", "senior", "new_member"]
            )],
            "claims": [f for f in self.feature_names if any(
                x in f for x in ["claim", "visit", "paid", "cost"]
            )],
            "diagnosis": [f for f in self.feature_names if any(
                x in f for x in ["has_", "diagnosis", "chronic", "complex", "risk"]
            )],
            "benefits": [f for f in self.feature_names if any(
                x in f for x in ["benefit", "utilization", "deductible", "remaining"]
            )],
            "engagement": [f for f in self.feature_names if any(
                x in f for x in ["engagement", "response", "session", "channel", 
                                "portal", "app", "email", "phone"]
            )],
            "temporal": [f for f in self.feature_names if any(
                x in f for x in ["month", "quarter", "season", "year_end", "urgency"]
            )]
        }


# COMMAND ----------

def create_member_features(
    members_df: pd.DataFrame,
    claims_df: pd.DataFrame,
    benefits_df: pd.DataFrame,
    engagement_df: pd.DataFrame,
    reference_date: Optional[datetime] = None
) -> Tuple[pd.DataFrame, FeatureEngineer]:
    """
    Main entry point for feature engineering
    
    Returns:
        Tuple of (features_df, feature_engineer)
    """
    engineer = FeatureEngineer(reference_date=reference_date)
    features = engineer.create_features(members_df, claims_df, benefits_df, engagement_df)
    return features, engineer

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample Usage
# MAGIC ```python
# MAGIC features_df, engineer = create_member_features(
# MAGIC     members_df, claims_df, benefits_df, engagement_df
# MAGIC )
# MAGIC 
# MAGIC # Get feature groups for analysis
# MAGIC feature_groups = engineer.get_feature_importance_groups()
# MAGIC ```

