# Databricks notebook source
# MAGIC %md
# MAGIC # Configuration Settings
# MAGIC Central configuration for the Healthcare Offer Prioritization project

# COMMAND ----------

from dataclasses import dataclass, field
from typing import List, Dict
import os

# COMMAND ----------

@dataclass
class DatabricksConfig:
    """Databricks-specific configuration"""
    # Unity Catalog settings
    catalog_name: str = "healthcare_demo"
    schema_name: str = "offer_prioritization"
    
    # Volume for artifacts
    volume_name: str = "artifacts"
    
    # MLflow settings
    experiment_name: str = "/Shared/healthcare_offer_prioritization"
    model_name: str = "healthcare_offer_prioritizer"
    
    @property
    def full_schema_path(self) -> str:
        return f"{self.catalog_name}.{self.schema_name}"
    
    @property
    def model_registry_path(self) -> str:
        return f"{self.catalog_name}.{self.schema_name}.{self.model_name}"


@dataclass
class DataConfig:
    """Data generation and schema configuration"""
    # Sample sizes
    num_members: int = 50000
    num_claims_per_member_range: tuple = (0, 25)
    num_engagements_per_member_range: tuple = (0, 50)
    
    # Table names
    members_table: str = "members"
    claims_table: str = "claims"
    benefits_table: str = "benefits_utilization"
    engagement_table: str = "engagement_history"
    features_table: str = "member_features"
    offers_table: str = "offer_catalog"
    scores_table: str = "offer_scores"
    
    # Date ranges
    claims_lookback_days: int = 730  # 2 years
    engagement_lookback_days: int = 365  # 1 year


@dataclass
class OfferConfig:
    """Offer catalog configuration"""
    offer_categories: List[Dict] = field(default_factory=lambda: [
        {
            "offer_id": "PREV_001",
            "name": "Annual Wellness Visit Reminder",
            "category": "preventive_care",
            "description": "Schedule your free annual wellness visit",
            "target_conditions": ["general"],
            "priority_weight": 1.2
        },
        {
            "offer_id": "PREV_002", 
            "name": "Cancer Screening Program",
            "category": "preventive_care",
            "description": "Age-appropriate cancer screenings covered at 100%",
            "target_conditions": ["general", "high_risk"],
            "priority_weight": 1.3
        },
        {
            "offer_id": "CHRON_001",
            "name": "Diabetes Management Program",
            "category": "chronic_disease",
            "description": "Comprehensive diabetes care and monitoring",
            "target_conditions": ["diabetes", "prediabetes"],
            "priority_weight": 1.5
        },
        {
            "offer_id": "CHRON_002",
            "name": "Heart Health Program",
            "category": "chronic_disease", 
            "description": "Cardiovascular disease prevention and management",
            "target_conditions": ["cardiovascular", "hypertension"],
            "priority_weight": 1.5
        },
        {
            "offer_id": "CHRON_003",
            "name": "Respiratory Care Program",
            "category": "chronic_disease",
            "description": "Asthma and COPD management support",
            "target_conditions": ["respiratory"],
            "priority_weight": 1.4
        },
        {
            "offer_id": "MH_001",
            "name": "Mental Health Support",
            "category": "mental_health",
            "description": "Access to counseling and therapy services",
            "target_conditions": ["mental_health", "stress"],
            "priority_weight": 1.3
        },
        {
            "offer_id": "MH_002",
            "name": "Stress Management Workshop",
            "category": "mental_health",
            "description": "Virtual stress reduction and mindfulness programs",
            "target_conditions": ["stress", "general"],
            "priority_weight": 1.1
        },
        {
            "offer_id": "PHARM_001",
            "name": "Generic Drug Switch Program",
            "category": "pharmacy",
            "description": "Save money by switching to generic medications",
            "target_conditions": ["high_rx_cost"],
            "priority_weight": 1.2
        },
        {
            "offer_id": "PHARM_002",
            "name": "Mail-Order Pharmacy",
            "category": "pharmacy",
            "description": "90-day supply with home delivery savings",
            "target_conditions": ["chronic_rx"],
            "priority_weight": 1.1
        },
        {
            "offer_id": "TELE_001",
            "name": "Virtual Primary Care",
            "category": "telehealth",
            "description": "24/7 access to virtual doctor visits",
            "target_conditions": ["general", "convenience"],
            "priority_weight": 1.0
        },
        {
            "offer_id": "TELE_002",
            "name": "Virtual Specialist Consultations",
            "category": "telehealth",
            "description": "Connect with specialists from home",
            "target_conditions": ["specialist_need"],
            "priority_weight": 1.2
        },
        {
            "offer_id": "FIT_001",
            "name": "Gym Membership Discount",
            "category": "fitness",
            "description": "50% off gym memberships nationwide",
            "target_conditions": ["fitness", "weight_management"],
            "priority_weight": 0.9
        },
        {
            "offer_id": "FIT_002",
            "name": "Nutrition Coaching",
            "category": "fitness",
            "description": "Personalized nutrition plans and coaching",
            "target_conditions": ["weight_management", "diabetes"],
            "priority_weight": 1.1
        },
        {
            "offer_id": "NAV_001",
            "name": "Care Navigator Assignment",
            "category": "care_navigation",
            "description": "Personal care coordinator for complex conditions",
            "target_conditions": ["complex_care", "multiple_chronic"],
            "priority_weight": 1.4
        },
        {
            "offer_id": "COST_001",
            "name": "HSA/FSA Optimization",
            "category": "cost_savings",
            "description": "Maximize your tax-advantaged health savings",
            "target_conditions": ["high_oop_cost"],
            "priority_weight": 1.0
        },
        {
            "offer_id": "COST_002",
            "name": "In-Network Provider Finder",
            "category": "cost_savings",
            "description": "Find quality in-network providers to reduce costs",
            "target_conditions": ["high_oop_cost", "oon_usage"],
            "priority_weight": 1.1
        }
    ])


@dataclass
class ModelConfig:
    """Model training configuration"""
    # Feature groups
    demographic_features: List[str] = field(default_factory=lambda: [
        "age", "gender_encoded", "region_encoded", "plan_type_encoded",
        "tenure_months", "income_bracket_encoded", "family_size"
    ])
    
    claims_features: List[str] = field(default_factory=lambda: [
        "total_claims_count", "total_claims_amount", "avg_claim_amount",
        "claims_last_90d", "claims_last_180d", "claims_last_365d",
        "er_visit_count", "inpatient_count", "outpatient_count",
        "preventive_visit_count", "specialist_visit_count",
        "days_since_last_claim", "claim_frequency_trend"
    ])
    
    diagnosis_features: List[str] = field(default_factory=lambda: [
        "has_diabetes", "has_cardiovascular", "has_respiratory",
        "has_mental_health", "has_musculoskeletal", "chronic_condition_count",
        "high_risk_flag"
    ])
    
    benefits_features: List[str] = field(default_factory=lambda: [
        "medical_utilization_rate", "pharmacy_utilization_rate",
        "preventive_utilization_rate", "dental_utilization_rate",
        "vision_utilization_rate", "mental_health_utilization_rate",
        "remaining_deductible_pct", "remaining_oop_max_pct"
    ])
    
    engagement_features: List[str] = field(default_factory=lambda: [
        "total_engagements", "email_engagement_rate", "app_engagement_rate",
        "call_engagement_rate", "portal_login_count", "days_since_last_engagement",
        "avg_response_rate", "preferred_channel_encoded"
    ])
    
    # Model hyperparameters
    lgbm_params: Dict = field(default_factory=lambda: {
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
        "early_stopping_rounds": 20
    })
    
    # Training settings
    test_size: float = 0.2
    random_state: int = 42
    cv_folds: int = 5


# COMMAND ----------

# Initialize default configurations
databricks_config = DatabricksConfig()
data_config = DataConfig()
offer_config = OfferConfig()
model_config = ModelConfig()

# COMMAND ----------

def get_table_path(table_name: str) -> str:
    """Get full table path in Unity Catalog"""
    return f"{databricks_config.full_schema_path}.{table_name}"


def print_config():
    """Print current configuration"""
    print("=" * 60)
    print("Healthcare Offer Prioritization - Configuration")
    print("=" * 60)
    print(f"\n📦 Catalog: {databricks_config.catalog_name}")
    print(f"📁 Schema: {databricks_config.schema_name}")
    print(f"🧪 Experiment: {databricks_config.experiment_name}")
    print(f"🤖 Model: {databricks_config.model_name}")
    print(f"\n👥 Members to generate: {data_config.num_members:,}")
    print(f"🏥 Offers in catalog: {len(offer_config.offer_categories)}")
    print("=" * 60)

