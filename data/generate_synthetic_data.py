# Databricks notebook source
# MAGIC %md
# MAGIC # Synthetic Data Generation
# MAGIC Generate realistic healthcare member data for the offer prioritization demo

# COMMAND ----------

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Tuple, List, Dict
import random
import uuid

# COMMAND ----------

# MAGIC %md
# MAGIC ## Configuration

# COMMAND ----------

class SyntheticDataGenerator:
    """Generate synthetic healthcare member data"""
    
    def __init__(self, seed: int = 42):
        np.random.seed(seed)
        random.seed(seed)
        self.seed = seed
        
        # Reference data
        self.regions = ["Northeast", "Southeast", "Midwest", "Southwest", "West"]
        self.plan_types = ["HMO", "PPO", "EPO", "HDHP", "POS"]
        self.income_brackets = ["<30K", "30K-50K", "50K-75K", "75K-100K", "100K-150K", ">150K"]
        self.genders = ["M", "F", "O"]
        
        self.claim_types = [
            "preventive", "primary_care", "specialist", "emergency", 
            "inpatient", "outpatient", "lab", "imaging", "procedure"
        ]
        
        self.diagnosis_categories = [
            "general_wellness", "cardiovascular", "diabetes", "respiratory",
            "mental_health", "musculoskeletal", "gastrointestinal", 
            "dermatological", "neurological", "oncology", "endocrine"
        ]
        
        self.benefit_types = [
            "medical", "pharmacy", "preventive", "dental", 
            "vision", "mental_health", "telehealth"
        ]
        
        self.engagement_channels = ["email", "app", "phone", "mail", "portal", "sms"]
        self.engagement_types = [
            "offer_view", "offer_click", "appointment_schedule",
            "content_view", "survey_complete", "portal_login", "app_open"
        ]
        
    def generate_members(self, n: int = 50000) -> pd.DataFrame:
        """Generate member demographics data"""
        
        # Age distribution weighted toward working age
        ages = np.random.choice(
            range(18, 85),
            size=n,
            p=self._age_distribution()
        )
        
        members = pd.DataFrame({
            "member_id": [f"MBR_{str(uuid.uuid4())[:8].upper()}" for _ in range(n)],
            "age": ages,
            "gender": np.random.choice(self.genders, n, p=[0.48, 0.50, 0.02]),
            "region": np.random.choice(self.regions, n),
            "plan_type": np.random.choice(self.plan_types, n, p=[0.25, 0.35, 0.10, 0.20, 0.10]),
            "tenure_months": np.random.exponential(36, n).astype(int).clip(1, 240),
            "income_bracket": np.random.choice(self.income_brackets, n, p=[0.10, 0.20, 0.25, 0.20, 0.15, 0.10]),
            "family_size": np.random.choice([1, 2, 3, 4, 5, 6], n, p=[0.25, 0.30, 0.20, 0.15, 0.07, 0.03]),
            "enrollment_date": [
                datetime.now() - timedelta(days=int(np.random.exponential(365*3))) 
                for _ in range(n)
            ],
            "risk_score": np.random.beta(2, 5, n) * 100,  # 0-100 scale, skewed low
            "created_at": datetime.now()
        })
        
        # Add chronic condition flags based on age and risk
        members["chronic_condition_count"] = self._generate_chronic_conditions(members)
        
        return members
    
    def generate_claims(self, members: pd.DataFrame, lookback_days: int = 730) -> pd.DataFrame:
        """Generate claims history for members"""
        
        claims_list = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        for _, member in members.iterrows():
            # Number of claims based on age, risk, and chronic conditions
            base_claims = np.random.poisson(3)
            age_factor = 1 + (member["age"] - 40) / 100
            risk_factor = 1 + member["risk_score"] / 100
            chronic_factor = 1 + member["chronic_condition_count"] * 0.3
            
            n_claims = int(base_claims * age_factor * risk_factor * chronic_factor)
            n_claims = max(0, min(n_claims, 50))  # Cap at 50 claims
            
            for _ in range(n_claims):
                claim_date = start_date + timedelta(
                    days=random.randint(0, lookback_days)
                )
                
                claim_type = self._weighted_claim_type(member)
                diagnosis = self._weighted_diagnosis(member, claim_type)
                amount = self._generate_claim_amount(claim_type)
                
                claims_list.append({
                    "claim_id": f"CLM_{str(uuid.uuid4())[:12].upper()}",
                    "member_id": member["member_id"],
                    "claim_date": claim_date,
                    "claim_type": claim_type,
                    "diagnosis_category": diagnosis,
                    "claim_amount": amount,
                    "allowed_amount": amount * np.random.uniform(0.7, 1.0),
                    "paid_amount": amount * np.random.uniform(0.5, 0.9),
                    "member_responsibility": amount * np.random.uniform(0.05, 0.3),
                    "provider_type": self._get_provider_type(claim_type),
                    "in_network": np.random.choice([True, False], p=[0.85, 0.15]),
                    "status": np.random.choice(
                        ["paid", "pending", "denied"], 
                        p=[0.90, 0.07, 0.03]
                    ),
                    "created_at": datetime.now()
                })
        
        return pd.DataFrame(claims_list)
    
    def generate_benefits_utilization(self, members: pd.DataFrame) -> pd.DataFrame:
        """Generate benefits utilization data for members"""
        
        benefits_list = []
        
        for _, member in members.iterrows():
            for benefit_type in self.benefit_types:
                # Utilization rate based on member characteristics
                base_rate = np.random.beta(2, 3)
                
                # Adjust based on benefit type and member profile
                if benefit_type == "preventive":
                    base_rate *= (1.2 if member["age"] > 50 else 0.9)
                elif benefit_type == "pharmacy":
                    base_rate *= (1 + member["chronic_condition_count"] * 0.2)
                elif benefit_type == "mental_health":
                    base_rate *= (1.3 if member["age"] < 40 else 0.8)
                
                utilization_rate = min(base_rate, 1.0)
                
                # Calculate remaining balance
                annual_max = self._get_benefit_max(benefit_type, member["plan_type"])
                used_amount = annual_max * utilization_rate
                
                last_used = None
                if utilization_rate > 0.1:
                    days_ago = int(np.random.exponential(60))
                    last_used = datetime.now() - timedelta(days=min(days_ago, 365))
                
                benefits_list.append({
                    "member_id": member["member_id"],
                    "benefit_type": benefit_type,
                    "annual_max": annual_max,
                    "used_amount": round(used_amount, 2),
                    "remaining_balance": round(annual_max - used_amount, 2),
                    "utilization_rate": round(utilization_rate, 4),
                    "last_used_date": last_used,
                    "claims_count": int(utilization_rate * np.random.randint(1, 20)),
                    "plan_year": datetime.now().year,
                    "created_at": datetime.now()
                })
        
        return pd.DataFrame(benefits_list)
    
    def generate_engagement_history(
        self, 
        members: pd.DataFrame, 
        lookback_days: int = 365
    ) -> pd.DataFrame:
        """Generate engagement history for members"""
        
        engagement_list = []
        end_date = datetime.now()
        start_date = end_date - timedelta(days=lookback_days)
        
        for _, member in members.iterrows():
            # Engagement frequency based on member characteristics
            base_engagements = np.random.poisson(10)
            
            # Younger members more digitally engaged
            age_factor = max(0.5, 1.5 - (member["age"] - 30) / 100)
            
            n_engagements = int(base_engagements * age_factor)
            n_engagements = max(0, min(n_engagements, 100))
            
            # Determine preferred channel based on age
            preferred_channel = self._get_preferred_channel(member["age"])
            
            for _ in range(n_engagements):
                engagement_date = start_date + timedelta(
                    days=random.randint(0, lookback_days)
                )
                
                # Weight toward preferred channel
                if random.random() < 0.6:
                    channel = preferred_channel
                else:
                    channel = random.choice(self.engagement_channels)
                
                engagement_type = self._get_engagement_type(channel)
                
                engagement_list.append({
                    "engagement_id": f"ENG_{str(uuid.uuid4())[:12].upper()}",
                    "member_id": member["member_id"],
                    "engagement_date": engagement_date,
                    "channel": channel,
                    "engagement_type": engagement_type,
                    "response_flag": np.random.choice([True, False], p=[0.3, 0.7]),
                    "session_duration_sec": int(np.random.exponential(120)) if channel in ["app", "portal"] else None,
                    "offer_id": f"OFFER_{random.randint(1, 16):03d}" if "offer" in engagement_type else None,
                    "created_at": datetime.now()
                })
        
        return pd.DataFrame(engagement_list)
    
    # Helper methods
    def _age_distribution(self) -> List[float]:
        """Create realistic age distribution"""
        ages = range(18, 85)
        weights = []
        for age in ages:
            if age < 26:
                w = 0.8
            elif age < 35:
                w = 1.2
            elif age < 50:
                w = 1.5
            elif age < 65:
                w = 1.3
            else:
                w = 0.7
            weights.append(w)
        total = sum(weights)
        return [w/total for w in weights]
    
    def _generate_chronic_conditions(self, members: pd.DataFrame) -> np.ndarray:
        """Generate chronic condition count based on age and risk"""
        conditions = np.zeros(len(members))
        
        for i, (_, member) in enumerate(members.iterrows()):
            # Base probability increases with age
            base_prob = min(0.1 + (member["age"] - 30) / 200, 0.5)
            
            # Add conditions based on probability
            n_conditions = 0
            for _ in range(5):  # Max 5 chronic conditions
                if random.random() < base_prob:
                    n_conditions += 1
                base_prob *= 0.5  # Decreasing probability for multiple
            
            conditions[i] = n_conditions
        
        return conditions.astype(int)
    
    def _weighted_claim_type(self, member: pd.Series) -> str:
        """Get claim type weighted by member characteristics"""
        weights = {
            "preventive": 0.15,
            "primary_care": 0.25,
            "specialist": 0.15,
            "emergency": 0.05,
            "inpatient": 0.03,
            "outpatient": 0.12,
            "lab": 0.10,
            "imaging": 0.08,
            "procedure": 0.07
        }
        
        # Adjust weights based on age
        if member["age"] > 60:
            weights["specialist"] *= 1.5
            weights["inpatient"] *= 1.5
        
        if member["chronic_condition_count"] > 0:
            weights["specialist"] *= 1.3
            weights["lab"] *= 1.2
        
        # Normalize
        total = sum(weights.values())
        probs = [w/total for w in weights.values()]
        
        return np.random.choice(list(weights.keys()), p=probs)
    
    def _weighted_diagnosis(self, member: pd.Series, claim_type: str) -> str:
        """Get diagnosis category based on member and claim type"""
        if claim_type == "preventive":
            return "general_wellness"
        
        # Base weights
        weights = {cat: 1.0 for cat in self.diagnosis_categories}
        
        # Adjust based on age
        if member["age"] > 50:
            weights["cardiovascular"] *= 2
            weights["musculoskeletal"] *= 1.5
        
        if member["age"] < 40:
            weights["mental_health"] *= 1.5
        
        # Normalize and select
        total = sum(weights.values())
        probs = [w/total for w in weights.values()]
        
        return np.random.choice(list(weights.keys()), p=probs)
    
    def _generate_claim_amount(self, claim_type: str) -> float:
        """Generate realistic claim amount based on type"""
        amount_ranges = {
            "preventive": (100, 500),
            "primary_care": (150, 400),
            "specialist": (200, 800),
            "emergency": (500, 5000),
            "inpatient": (5000, 50000),
            "outpatient": (500, 3000),
            "lab": (50, 500),
            "imaging": (200, 2000),
            "procedure": (1000, 10000)
        }
        
        low, high = amount_ranges.get(claim_type, (100, 1000))
        return round(np.random.lognormal(np.log((low + high) / 2), 0.5), 2)
    
    def _get_provider_type(self, claim_type: str) -> str:
        """Get provider type based on claim type"""
        mapping = {
            "preventive": "primary_care",
            "primary_care": "primary_care",
            "specialist": "specialist",
            "emergency": "emergency_room",
            "inpatient": "hospital",
            "outpatient": "ambulatory",
            "lab": "laboratory",
            "imaging": "imaging_center",
            "procedure": "ambulatory_surgical"
        }
        return mapping.get(claim_type, "other")
    
    def _get_benefit_max(self, benefit_type: str, plan_type: str) -> float:
        """Get annual benefit maximum"""
        base_amounts = {
            "medical": 100000,
            "pharmacy": 5000,
            "preventive": 2000,
            "dental": 2000,
            "vision": 500,
            "mental_health": 5000,
            "telehealth": 1000
        }
        
        plan_multipliers = {
            "HMO": 0.9,
            "PPO": 1.1,
            "EPO": 1.0,
            "HDHP": 0.8,
            "POS": 1.0
        }
        
        base = base_amounts.get(benefit_type, 1000)
        multiplier = plan_multipliers.get(plan_type, 1.0)
        
        return base * multiplier
    
    def _get_preferred_channel(self, age: int) -> str:
        """Get preferred engagement channel based on age"""
        if age < 30:
            return np.random.choice(["app", "sms"], p=[0.6, 0.4])
        elif age < 50:
            return np.random.choice(["app", "email", "portal"], p=[0.4, 0.35, 0.25])
        elif age < 65:
            return np.random.choice(["email", "portal", "phone"], p=[0.4, 0.3, 0.3])
        else:
            return np.random.choice(["phone", "mail"], p=[0.6, 0.4])
    
    def _get_engagement_type(self, channel: str) -> str:
        """Get engagement type based on channel"""
        channel_types = {
            "app": ["app_open", "offer_view", "offer_click", "content_view"],
            "portal": ["portal_login", "offer_view", "content_view", "appointment_schedule"],
            "email": ["offer_view", "offer_click", "survey_complete"],
            "sms": ["offer_view", "offer_click", "appointment_schedule"],
            "phone": ["appointment_schedule", "survey_complete"],
            "mail": ["offer_view", "survey_complete"]
        }
        
        return random.choice(channel_types.get(channel, ["offer_view"]))


# COMMAND ----------

def generate_all_data(
    n_members: int = 50000,
    claims_lookback: int = 730,
    engagement_lookback: int = 365,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Generate all synthetic data for the demo
    
    Returns:
        Tuple of (members_df, claims_df, benefits_df, engagement_df)
    """
    print("🏥 Generating Healthcare Offer Prioritization Demo Data")
    print("=" * 60)
    
    generator = SyntheticDataGenerator(seed=seed)
    
    print(f"\n👥 Generating {n_members:,} members...")
    members_df = generator.generate_members(n_members)
    print(f"   ✓ Created {len(members_df):,} member records")
    
    print(f"\n📋 Generating claims (last {claims_lookback} days)...")
    claims_df = generator.generate_claims(members_df, claims_lookback)
    print(f"   ✓ Created {len(claims_df):,} claim records")
    
    print(f"\n💊 Generating benefits utilization...")
    benefits_df = generator.generate_benefits_utilization(members_df)
    print(f"   ✓ Created {len(benefits_df):,} benefit records")
    
    print(f"\n📱 Generating engagement history (last {engagement_lookback} days)...")
    engagement_df = generator.generate_engagement_history(members_df, engagement_lookback)
    print(f"   ✓ Created {len(engagement_df):,} engagement records")
    
    print("\n" + "=" * 60)
    print("✅ Data generation complete!")
    
    return members_df, claims_df, benefits_df, engagement_df

# COMMAND ----------

# MAGIC %md
# MAGIC ## Sample Usage
# MAGIC ```python
# MAGIC # Generate data
# MAGIC members, claims, benefits, engagement = generate_all_data(n_members=10000)
# MAGIC 
# MAGIC # Save to Delta tables
# MAGIC members.to_spark().write.mode("overwrite").saveAsTable("catalog.schema.members")
# MAGIC ```

