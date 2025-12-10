# Healthcare Offer Prioritization System

An end-to-end machine learning system built on Databricks that personalizes healthcare offers for insurance members. The system analyzes member demographics, claims history, benefits utilization, and engagement patterns to rank and recommend the most relevant healthcare programs for each individual.

---

## Table of Contents

1. [Overview](#overview)
2. [How It Works](#how-it-works)
3. [Architecture](#architecture)
4. [Project Structure](#project-structure)
5. [Data Schema](#data-schema)
6. [Offer Catalog](#offer-catalog)
7. [Feature Engineering](#feature-engineering)
8. [Model Details](#model-details)
9. [Getting Started](#getting-started)
10. [Running the Notebooks](#running-the-notebooks)
11. [LLM-Powered Explanations](#llm-powered-explanations)
12. [Configuration](#configuration)
13. [Troubleshooting](#troubleshooting)

---

## Overview

### What This Project Does

This system solves the problem of **offer fatigue** in healthcare marketing. Instead of sending the same generic offers to all members, it:

1. **Analyzes** each member's health profile, claims patterns, and engagement history
2. **Predicts** which healthcare programs would be most beneficial for each member
3. **Ranks** offers by priority score (0-100) based on relevance
4. **Explains** why each offer was recommended using SHAP values + LLM summarization

### Key Features

| Feature | Description |
|---------|-------------|
| 🎯 **Personalized Scoring** | Each member gets unique priority scores for all 16 healthcare offers |
| 📊 **Multi-Output ML Model** | Single LightGBM model predicts scores for all offers simultaneously |
| 🔍 **Explainability** | SHAP values identify which features drove each recommendation |
| 🤖 **LLM Reasoning** | Natural language explanations generated for each recommendation |
| 📦 **MLflow Integration** | Full experiment tracking, model registry, and versioning |
| ⚡ **Batch & Real-time** | Supports both batch scoring and real-time inference |

---

## How It Works

### The Pipeline

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│   Raw Data      │     │    Features     │     │     Model       │     │  Recommendations│
│                 │     │                 │     │                 │     │                 │
│ • Members       │────▶│ • Demographics  │────▶│ • LightGBM      │────▶│ • Top 5 offers  │
│ • Claims        │     │ • Claims aggs   │     │ • Multi-output  │     │ • Priority scores│
│ • Benefits      │     │ • Diagnosis flags│     │ • 16 targets    │     │ • SHAP values   │
│ • Engagement    │     │ • Benefits util │     │   (one per offer)│     │ • LLM reasoning │
│                 │     │ • Engagement    │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Step-by-Step Flow

1. **Data Ingestion** (Notebook 01)
   - Load member demographics, claims, benefits, and engagement data
   - In demo mode, generates realistic synthetic data

2. **Feature Engineering** (Notebook 02)
   - Creates 80+ features from raw data
   - Aggregates claims by time windows, diagnosis categories
   - Encodes categorical variables
   - Creates behavioral and temporal features

3. **Model Training** (Notebook 03)
   - Generates target scores using business rules
   - Trains LightGBM multi-output regressor
   - Logs to MLflow with metrics, artifacts, and model signature
   - Registers model in Unity Catalog

4. **Inference & Explanation** (Notebook 04)
   - Loads model from Unity Catalog (or trains fresh for demo)
   - Generates recommendations for all members
   - Computes SHAP values for explainability
   - Calls LLM to generate human-readable reasoning

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────────┐
│                         DATABRICKS WORKSPACE                           │
├────────────────────────────────────────────────────────────────────────┤
│                                                                        │
│  ┌──────────────┐    ┌──────────────┐    ┌──────────────┐             │
│  │  Notebook 01 │    │  Notebook 02 │    │  Notebook 03 │             │
│  │  Data Explore│───▶│  Features    │───▶│  Training    │             │
│  └──────────────┘    └──────────────┘    └──────┬───────┘             │
│                                                  │                     │
│                                                  ▼                     │
│  ┌──────────────────────────────────────────────────────────┐         │
│  │                    MLFLOW                                 │         │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐      │         │
│  │  │ Experiments │  │  Artifacts  │  │  Registry   │      │         │
│  │  │ • Params    │  │ • Model     │  │ • Versions  │      │         │
│  │  │ • Metrics   │  │ • SHAP plot │  │ • Aliases   │      │         │
│  │  │ • Tags      │  │ • Metadata  │  │ • Stage     │      │         │
│  │  └─────────────┘  └─────────────┘  └──────┬──────┘      │         │
│  └───────────────────────────────────────────┼──────────────┘         │
│                                               │                        │
│                                               ▼                        │
│  ┌──────────────┐    ┌──────────────────────────────────────┐         │
│  │  Notebook 04 │◀───│  Unity Catalog Model Registry        │         │
│  │  Inference   │    │  healthcare_demo.offer_prioritization│         │
│  └──────┬───────┘    │  .healthcare_offer_prioritizer       │         │
│         │            └──────────────────────────────────────┘         │
│         ▼                                                              │
│  ┌──────────────┐    ┌──────────────┐                                 │
│  │    SHAP      │───▶│  Foundation  │                                 │
│  │  Explainer   │    │  Model API   │                                 │
│  │              │    │  (LLaMA 3.1) │                                 │
│  └──────────────┘    └──────┬───────┘                                 │
│                              │                                         │
│                              ▼                                         │
│  ┌──────────────────────────────────────────────────────────┐         │
│  │                    OUTPUT                                 │         │
│  │  • Personalized offer rankings per member                │         │
│  │  • Priority scores (0-100)                               │         │
│  │  • Feature importance per recommendation                 │         │
│  │  • Natural language reasoning                            │         │
│  └──────────────────────────────────────────────────────────┘         │
│                                                                        │
└────────────────────────────────────────────────────────────────────────┘
```

---

## Project Structure

```
offer_prioritization/
│
├── 📁 config/
│   ├── __init__.py
│   └── config.py                 # Central configuration (catalog, schema, model params)
│
├── 📁 data/
│   ├── __init__.py
│   └── generate_synthetic_data.py  # Generates realistic synthetic healthcare data
│
├── 📁 features/
│   ├── __init__.py
│   └── feature_engineering.py    # FeatureEngineer class - creates 80+ ML features
│
├── 📁 models/
│   ├── __init__.py
│   └── offer_model.py            # OfferPrioritizationModel, OfferCatalog, RuleBasedScorer
│
├── 📁 notebooks/
│   ├── __init__.py
│   ├── 01_data_exploration.py    # Data loading and exploratory analysis
│   ├── 02_feature_engineering.py # Feature creation and analysis
│   ├── 03_model_training.py      # Model training with MLflow tracking
│   └── 04_model_inference.py     # Inference, SHAP explanations, LLM reasoning
│
├── 📁 utils/
│   ├── __init__.py
│   └── helpers.py                # Utility functions
│
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
├── run_demo.py                   # Quick demo runner script
└── README.md                     # This file
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| **FeatureEngineer** | `features/feature_engineering.py` | Transforms raw data into ML features |
| **OfferCatalog** | `models/offer_model.py` | Defines the 16 healthcare offers |
| **RuleBasedScorer** | `models/offer_model.py` | Generates training labels from business rules |
| **OfferPrioritizationModel** | `models/offer_model.py` | LightGBM multi-output wrapper |
| **OfferRecommendationEngine** | `notebooks/04_model_inference.py` | Generates recommendations with filters |

---

## Data Schema

### Members Table

| Column | Type | Description |
|--------|------|-------------|
| `member_id` | string | Unique member identifier (e.g., "M00001") |
| `age` | int | Member age (18-85) |
| `gender` | string | "M", "F", or "Other" |
| `region` | string | Geographic region (Northeast, Southeast, etc.) |
| `plan_type` | string | Insurance plan (HMO, PPO, EPO, HDHP) |
| `tenure_months` | int | Months as a member (1-240) |
| `income_bracket` | string | Income level (Low, Medium, High, Very High) |
| `family_size` | int | Number of family members (1-6) |
| `risk_score` | float | Health risk score (0-100) |
| `chronic_condition_count` | int | Number of chronic conditions (0-5) |

### Claims Table

| Column | Type | Description |
|--------|------|-------------|
| `claim_id` | string | Unique claim identifier |
| `member_id` | string | Foreign key to members |
| `claim_date` | date | Date of service |
| `claim_type` | string | Type: primary_care, specialist, emergency, etc. |
| `claim_amount` | float | Total billed amount |
| `paid_amount` | float | Amount paid by insurance |
| `member_responsibility` | float | Member's out-of-pocket cost |
| `diagnosis_category` | string | Primary diagnosis category |
| `provider_type` | string | Type of provider |
| `in_network` | bool | Whether provider was in-network |

### Benefits Utilization Table

| Column | Type | Description |
|--------|------|-------------|
| `member_id` | string | Foreign key to members |
| `benefit_type` | string | medical, pharmacy, dental, vision, mental_health, preventive |
| `annual_max` | float | Annual benefit maximum |
| `used_amount` | float | Amount used YTD |
| `remaining_balance` | float | Remaining benefit amount |
| `utilization_rate` | float | Percentage used (0-1) |
| `claims_count` | int | Number of claims for this benefit |
| `last_used_date` | date | Most recent usage date |

### Engagement History Table

| Column | Type | Description |
|--------|------|-------------|
| `engagement_id` | string | Unique engagement identifier |
| `member_id` | string | Foreign key to members |
| `engagement_date` | date | Date of engagement |
| `channel` | string | email, app, portal, phone, mail |
| `engagement_type` | string | offer_sent, offer_opened, offer_clicked, etc. |
| `response_flag` | bool | Whether member responded |
| `session_duration_sec` | int | Duration if applicable |

---

## Offer Catalog

The system prioritizes 16 healthcare offers across 8 categories:

| Category | Offers | Target Members |
|----------|--------|----------------|
| **Preventive Care** | Annual Wellness Visit, Cancer Screening | Members overdue for checkups, age 50+ |
| **Chronic Disease** | Diabetes Management, Heart Health, Respiratory Care | Members with specific conditions |
| **Mental Health** | Mental Health Support, Stress Management | Members with MH history, high utilizers |
| **Pharmacy** | Generic Drug Switch, Mail-Order Pharmacy | High pharmacy utilization, chronic Rx |
| **Telehealth** | Virtual Primary Care, Virtual Specialists | App users, complex patients |
| **Fitness** | Gym Discount, Nutrition Coaching | Younger members, weight management |
| **Care Navigation** | Care Navigator Assignment | Complex patients, multiple conditions |
| **Cost Savings** | HSA/FSA Optimization, In-Network Finder | High OOP costs, out-of-network usage |

---

## Feature Engineering

The `FeatureEngineer` class creates **80+ features** organized into groups:

### Feature Groups

| Group | # Features | Examples |
|-------|------------|----------|
| **Demographic** | 15 | age, tenure_months, risk_score, is_senior, age_group_encoded |
| **Claims** | 20 | total_claims_count, avg_claim_amount, claims_last_90d, er_visit_count |
| **Diagnosis** | 10 | has_diabetes, has_cardiovascular, is_complex_patient |
| **Benefits** | 12 | avg_utilization_rate, pharmacy_utilization_rate, remaining_deductible_pct |
| **Engagement** | 15 | total_engagements, avg_response_rate, app_engagement_rate |
| **Temporal** | 8 | is_q4, is_flu_season, days_until_year_end, benefits_urgency |

### Feature Creation Example

```python
from features.feature_engineering import create_member_features

# Create all features from raw data
features_df, feature_engineer = create_member_features(
    members_df=members_df,
    claims_df=claims_df,
    benefits_df=benefits_df,
    engagement_df=engagement_df,
    reference_date=datetime.now()
)

# Get feature groups for analysis
feature_groups = feature_engineer.get_feature_importance_groups()
```

---

## Model Details

### Architecture

```
                    ┌─────────────────────────────────────┐
                    │         Input Features (80+)        │
                    │  [age, claims, benefits, engage...] │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
                    ┌─────────────────────────────────────┐
                    │         StandardScaler              │
                    │     (normalize feature scales)      │
                    └─────────────────┬───────────────────┘
                                      │
                                      ▼
┌──────────────────────────────────────────────────────────────────────────┐
│                     MultiOutputRegressor                                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐      ┌─────────────┐   │
│  │  LightGBM   │ │  LightGBM   │ │  LightGBM   │ ...  │  LightGBM   │   │
│  │  Offer 1    │ │  Offer 2    │ │  Offer 3    │      │  Offer 16   │   │
│  │  (PREV_001) │ │  (PREV_002) │ │  (CHRON_001)│      │  (COST_002) │   │
│  └──────┬──────┘ └──────┬──────┘ └──────┬──────┘      └──────┬──────┘   │
│         │               │               │                    │          │
└─────────┼───────────────┼───────────────┼────────────────────┼──────────┘
          │               │               │                    │
          ▼               ▼               ▼                    ▼
    ┌──────────┐    ┌──────────┐    ┌──────────┐         ┌──────────┐
    │ Score 1  │    │ Score 2  │    │ Score 3  │   ...   │ Score 16 │
    │  (0-100) │    │  (0-100) │    │  (0-100) │         │  (0-100) │
    └──────────┘    └──────────┘    └──────────┘         └──────────┘
```

### Training Target Generation

Training labels are generated using `RuleBasedScorer` which applies business rules:

```python
# Example: Diabetes Management Program scoring
base_score = 50
if member.has_diabetes:
    score += 40  # Strong signal
if member.high_risk_flag:
    score += 10  # Additional boost
if member.pharmacy_utilization_rate > 0.3:
    score += 5   # Uses pharmacy benefits
```

### Hyperparameters

| Parameter | Value | Description |
|-----------|-------|-------------|
| `n_estimators` | 200 | Number of boosting rounds |
| `learning_rate` | 0.05 | Step size shrinkage |
| `num_leaves` | 31 | Max leaves per tree |
| `feature_fraction` | 0.8 | Features per tree |
| `bagging_fraction` | 0.8 | Data per tree |

---

## Getting Started

### Prerequisites

- Databricks workspace with:
  - Unity Catalog enabled
  - MLflow enabled
  - Foundation Model API access (for LLM reasoning)
- Python 3.9+

### Installation

1. **Clone to Databricks Repos**
   ```
   Repos → Add Repo → paste this repository URL
   ```

2. **Create a cluster** with:
   - Databricks Runtime 13.3 LTS ML or higher
   - Install additional libraries:
   ```bash
     %pip install shap openai
     ```

3. **Set up Unity Catalog** (optional, for model registry):
   ```sql
   CREATE CATALOG IF NOT EXISTS healthcare_demo;
   CREATE SCHEMA IF NOT EXISTS healthcare_demo.offer_prioritization;
   ```

### Quick Start

Run the notebooks in order:

```
01_data_exploration.py    →  Understand the data
02_feature_engineering.py →  Create ML features
03_model_training.py      →  Train and register model
04_model_inference.py     →  Generate recommendations
```

---

## Running the Notebooks

### Notebook 01: Data Exploration

**Purpose:** Load and explore member data

**Key Outputs:**
- Data distributions and statistics
- Correlation analysis
- Missing value assessment

**Time:** ~2 minutes

---

### Notebook 02: Feature Engineering

**Purpose:** Transform raw data into ML features

**Key Outputs:**
- `features_df` - DataFrame with 80+ features per member
- `feature_engineer` - Fitted transformer with encoders

**Time:** ~5 minutes (50K members)

---

### Notebook 03: Model Training

**Purpose:** Train model and log to MLflow

**Key Outputs:**
- Trained `OfferPrioritizationModel`
- MLflow run with metrics, artifacts
- Registered model in Unity Catalog

**Metrics Tracked:**
- RMSE (train/test)
- MAE (train/test)
- R² score
- Per-offer RMSE

**Time:** ~10 minutes

---

### Notebook 04: Model Inference

**Purpose:** Generate and explain recommendations

**Configuration:**
```python
# Set to True to load from Unity Catalog
USE_REGISTRY_MODEL = True

# Model version or alias
UC_MODEL_VERSION = "1"
UC_MODEL_ALIAS = None  # or "champion"
```

**Key Outputs:**
- `all_recommendations` - DataFrame with ranked offers per member
- `explanations_df` - SHAP-based feature importance per recommendation
- `reasoned_explanations` - LLM-generated reasoning

**Time:** ~15 minutes (10K members with SHAP + LLM)

---

## LLM-Powered Explanations

The system generates natural language explanations for each recommendation.

### How It Works

1. **SHAP Analysis** - Computes feature contributions for each member-offer pair
2. **Context Building** - Combines member profile + top contributing features
3. **LLM Generation** - Calls Foundation Model API to generate 3-4 sentence explanation

### Example Output

```
👤 MEMBER: M12345
   Age: 58 | Risk Score: 72.3
────────────────────────────────────────

📋 [1] Diabetes Management Program
   Score: 87.5/100

💬 Why this offer?
   Given your diabetes diagnosis and elevated health risk indicators, 
   our Diabetes Management Program is specifically designed to support 
   your wellness journey. This program offers personalized coaching, 
   medication management guidance, and regular check-ins to help you 
   maintain stable blood sugar levels and prevent complications. With 
   your established pattern of proactive healthcare engagement, you're 
   well-positioned to benefit from the comprehensive support this 
   program provides.
```

### Configuration

```python
# Set your LLM endpoint
LLM_ENDPOINT_NAME = "databricks-meta-llama-3-1-70b-instruct"

# Or use a custom model serving endpoint
LLM_ENDPOINT_NAME = "your-custom-endpoint"
```

---

## Configuration

### Main Configuration File: `config/config.py`

```python
@dataclass
class DatabricksConfig:
    catalog_name: str = "healthcare_demo"
    schema_name: str = "offer_prioritization"
    experiment_name: str = "/Shared/healthcare_offer_prioritization"
    model_name: str = "healthcare_offer_prioritizer"
```

### Environment Variables (for LLM)

| Variable | Purpose |
|----------|---------|
| `OPENAI_API_KEY` | API key for external LLM (if not using Databricks) |
| `OPENAI_BASE_URL` | Custom API endpoint URL |
| `DATABRICKS_TOKEN` | Databricks PAT (auto-set in workspace) |

---

## Troubleshooting

### Common Issues

#### 1. `AttributeError: 'bool' object has no attribute 'astype'`

**Cause:** Bug in temporal feature creation
**Solution:** Already fixed in `feature_engineering.py`

#### 2. `ValueError: could not convert string to float: '65+'`

**Cause:** String categorical columns not being encoded
**Solution:** Already fixed - `age_group` and `tenure_group` now properly encoded

#### 3. `MlflowException: Method 'get_latest_versions' is unsupported for Unity Catalog`

**Cause:** Unity Catalog uses aliases, not "latest"
**Solution:** Use version number or alias:
```python
load_model_from_registry(MODEL_NAME, version="1")
# or
load_model_from_registry(MODEL_NAME, alias="champion")
```

#### 4. `TypeError: predict() got an unexpected keyword argument 'return_top_n'`

**Cause:** Loading raw sklearn model instead of wrapper
**Solution:** Use `wrap_loaded_model()` or load the joblib artifact

#### 5. `Connection error` when generating LLM reasoning

**Cause:** Foundation Model API not accessible
**Solutions:**
- Check endpoint name: `databricks-meta-llama-3-1-70b-instruct`
- Verify Foundation Model API is enabled in your workspace
- Check user permissions for serving endpoints

### Getting Help

1. Check the error message in the notebook output
2. Review the troubleshooting section above
3. Check MLflow experiment logs for training issues
4. Verify Unity Catalog permissions for model registry issues

---

## License

MIT License - see LICENSE file for details.

---

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## Authors

Healthcare Data Science Team

---

*Last Updated: December 2024*
