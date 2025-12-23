import dash
from dash import html, dcc, Input, Output, State, callback_context
import os
import json
from databricks.sdk import WorkspaceClient

# Configuration - UPDATE THESE TO MATCH YOUR TABLE
CATALOG = "demos"
SCHEMA = "offer_prioritization"
TABLE = "member_offer_recommendations_with_reasoning"
FULL_TABLE_NAME = f"{CATALOG}.{SCHEMA}.{TABLE}"

# Initialize workspace client (auto-configured in Databricks Apps)
workspace_client = WorkspaceClient()

def execute_query(query: str, params: tuple = None):
    """
    Execute a SQL query using Databricks SDK Statement Execution API.
    This works automatically in Databricks Apps without manual connection setup.
    """
    try:
        # Use the Statement Execution API via workspace client
        # This automatically uses the app's credentials
        response = workspace_client.statement_execution.execute_statement(
            warehouse_id=os.getenv("DATABRICKS_WAREHOUSE_ID"),
            catalog=CATALOG,
            schema=SCHEMA,
            statement=query,
            wait_timeout="30s"
        )
        
        if response.result and response.result.data_array:
            columns = [col.name for col in response.manifest.schema.columns]
            return [dict(zip(columns, row)) for row in response.result.data_array]
        return []
    except Exception as e:
        print(f"Query error: {e}")
        # Fallback: try using spark SQL if available
        try:
            from pyspark.sql import SparkSession
            spark = SparkSession.builder.getOrCreate()
            if params:
                # Replace ? with actual values for spark
                for param in params:
                    query = query.replace("?", f"'{param}'", 1)
            df = spark.sql(query)
            return [row.asDict() for row in df.collect()]
        except Exception as e2:
            print(f"Spark fallback error: {e2}")
            return []

def search_members(search_term: str, limit: int = 20):
    """Search for members by member_id."""
    query = f"""
        SELECT DISTINCT member_id 
        FROM {FULL_TABLE_NAME}
        WHERE member_id LIKE '%{search_term}%'
        ORDER BY member_id
        LIMIT {limit}
    """
    results = execute_query(query)
    return [row['member_id'] for row in results]

def get_member_recommendations(member_id: str):
    """Get all recommendations for a specific member."""
    query = f"""
        SELECT 
            member_id,
            offer_id,
            offer_name,
            rank,
            priority_score,
            llm_reasoning,
            shap_factors,
            age,
            risk_score,
            chronic_condition_count,
            has_diabetes,
            has_cardiovascular,
            has_respiratory,
            has_mental_health,
            is_complex_patient,
            tenure_months,
            total_claims,
            total_engagements,
            avg_utilization_rate,
            high_risk_flag
        FROM {FULL_TABLE_NAME}
        WHERE member_id = '{member_id}'
        ORDER BY rank
        LIMIT 5
    """
    return execute_query(query)

def get_all_members():
    """Get all unique member IDs for dropdown."""
    query = f"""
        SELECT DISTINCT member_id 
        FROM {FULL_TABLE_NAME}
        ORDER BY member_id
        LIMIT 500
    """
    results = execute_query(query)
    return [row['member_id'] for row in results]

FEEDBACK_TABLE_CREATED = False

def ensure_feedback_table():
    """Create the feedback table if it doesn't exist."""
    global FEEDBACK_TABLE_CREATED
    if FEEDBACK_TABLE_CREATED:
        return True
    
    try:
        create_query = f"""
            CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA}.offer_feedback (
                member_id STRING,
                offer_id STRING,
                feedback STRING,
                feedback_text STRING,
                feedback_time TIMESTAMP
            )
        """
        execute_query(create_query)
        FEEDBACK_TABLE_CREATED = True
        print(f"✅ Feedback table ready: {CATALOG}.{SCHEMA}.offer_feedback")
        return True
    except Exception as e:
        print(f"⚠️ Could not create feedback table: {e}")
        return False

def save_feedback(member_id: str, offer_id: str, feedback: str, feedback_text: str = None):
    """Save offer feedback. Logs feedback and saves to database."""
    text_display = f" - Comment: {feedback_text[:50]}..." if feedback_text and len(feedback_text) > 50 else (f" - Comment: {feedback_text}" if feedback_text else "")
    print(f"📝 Feedback received: {member_id} - {offer_id} - {feedback}{text_display}")
    
    # Ensure table exists
    ensure_feedback_table()
    
    try:
        # Escape single quotes in text feedback
        safe_text = feedback_text.replace("'", "''") if feedback_text else ""
        query = f"INSERT INTO {CATALOG}.{SCHEMA}.offer_feedback (member_id, offer_id, feedback, feedback_text, feedback_time) VALUES ('{member_id}', '{offer_id}', '{feedback}', '{safe_text}', current_timestamp())"
        execute_query(query)
        print(f"✅ Saved to database: {CATALOG}.{SCHEMA}.offer_feedback")
        return True
    except Exception as e:
        print(f"⚠️ Could not save feedback to DB: {e}")
        return False

# Cache member list on startup
MEMBER_LIST = []
INIT_ERROR = None

def initialize_member_list():
    """Initialize the member list cache."""
    global MEMBER_LIST, INIT_ERROR
    try:
        MEMBER_LIST = get_all_members()
        print(f"✅ Loaded {len(MEMBER_LIST)} members into cache")
        if len(MEMBER_LIST) == 0:
            INIT_ERROR = "No members found in table. Check if table exists and has data."
        return MEMBER_LIST
    except Exception as e:
        INIT_ERROR = f"Failed to load members: {str(e)}"
        print(f"❌ {INIT_ERROR}")
        return []

# Initialize Dash app
app = dash.Dash(__name__)

# Custom CSS styles
styles = {
    'container': {
        'maxWidth': '1400px',
        'margin': '0 auto',
        'padding': '20px 40px',
        'fontFamily': '-apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    },
    'header': {
        'textAlign': 'center',
        'marginBottom': '30px',
        'color': '#1a365d'
    },
    'search_box': {
        'display': 'block',
        'marginBottom': '30px',
        'padding': '24px',
        'backgroundColor': '#f7fafc',
        'borderRadius': '12px',
        'boxShadow': '0 2px 4px rgba(0,0,0,0.1)'
    },
    'input': {
        'flex': '1',
        'padding': '12px 16px',
        'fontSize': '16px',
        'border': '2px solid #e2e8f0',
        'borderRadius': '8px',
        'outline': 'none'
    },
    'button': {
        'padding': '12px 24px',
        'backgroundColor': '#3182ce',
        'color': 'white',
        'border': 'none',
        'borderRadius': '8px',
        'cursor': 'pointer',
        'fontSize': '16px',
        'fontWeight': '600'
    },
    'member_card': {
        'backgroundColor': 'white',
        'borderRadius': '12px',
        'padding': '24px',
        'marginBottom': '20px',
        'boxShadow': '0 4px 6px rgba(0,0,0,0.1)',
        'border': '1px solid #e2e8f0'
    },
    'profile_section': {
        'display': 'grid',
        'gridTemplateColumns': 'repeat(auto-fit, minmax(150px, 1fr))',
        'gap': '16px',
        'marginBottom': '24px',
        'padding': '16px',
        'backgroundColor': '#f7fafc',
        'borderRadius': '8px'
    },
    'profile_item': {
        'textAlign': 'center'
    },
    'profile_label': {
        'fontSize': '12px',
        'color': '#718096',
        'textTransform': 'uppercase',
        'marginBottom': '4px'
    },
    'profile_value': {
        'fontSize': '20px',
        'fontWeight': '700',
        'color': '#2d3748'
    },
    'offer_card': {
        'backgroundColor': '#ffffff',
        'border': '1px solid #e2e8f0',
        'borderRadius': '10px',
        'padding': '20px',
        'marginBottom': '16px'
    },
    'offer_header': {
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'center',
        'marginBottom': '12px'
    },
    'rank_badge': {
        'backgroundColor': '#3182ce',
        'color': 'white',
        'padding': '4px 12px',
        'borderRadius': '20px',
        'fontSize': '14px',
        'fontWeight': '600'
    },
    'score_badge': {
        'backgroundColor': '#48bb78',
        'color': 'white',
        'padding': '4px 12px',
        'borderRadius': '20px',
        'fontSize': '14px',
        'fontWeight': '600'
    },
    'reasoning_box': {
        'backgroundColor': '#f0fff4',
        'border': '1px solid #9ae6b4',
        'borderRadius': '8px',
        'padding': '16px',
        'marginTop': '12px',
        'marginBottom': '12px'
    },
    'shap_box': {
        'backgroundColor': '#f8fafc',
        'border': '1px solid #e2e8f0',
        'borderRadius': '8px',
        'padding': '20px',
        'marginTop': '12px'
    },
    'shap_item': {
        'display': 'flex',
        'alignItems': 'flex-start',
        'padding': '12px 0',
        'borderBottom': '1px solid #edf2f7',
        'gap': '12px'
    },
    'shap_item_icon': {
        'fontSize': '20px',
        'lineHeight': '1',
        'marginTop': '2px'
    },
    'shap_item_content': {
        'flex': '1'
    },
    'shap_item_title': {
        'fontWeight': '600',
        'color': '#2d3748',
        'marginBottom': '4px',
        'fontSize': '15px'
    },
    'shap_item_description': {
        'color': '#718096',
        'fontSize': '14px',
        'lineHeight': '1.5'
    },
    'explain_button': {
        'padding': '10px 20px',
        'backgroundColor': '#edf2f7',
        'color': '#4a5568',
        'border': '1px solid #e2e8f0',
        'borderRadius': '8px',
        'cursor': 'pointer',
        'fontSize': '14px',
        'fontWeight': '600',
        'display': 'flex',
        'alignItems': 'center',
        'gap': '8px',
        'marginTop': '12px',
        'transition': 'all 0.2s ease'
    },
    'explain_button_active': {
        'padding': '10px 20px',
        'backgroundColor': '#3182ce',
        'color': 'white',
        'border': '1px solid #3182ce',
        'borderRadius': '8px',
        'cursor': 'pointer',
        'fontSize': '14px',
        'fontWeight': '600',
        'display': 'flex',
        'alignItems': 'center',
        'gap': '8px',
        'marginTop': '12px'
    },
    'impact_badge_positive': {
        'display': 'inline-block',
        'backgroundColor': '#c6f6d5',
        'color': '#22543d',
        'padding': '2px 8px',
        'borderRadius': '12px',
        'fontSize': '12px',
        'fontWeight': '600',
        'marginLeft': '8px'
    },
    'impact_badge_negative': {
        'display': 'inline-block',
        'backgroundColor': '#fed7d7',
        'color': '#742a2a',
        'padding': '2px 8px',
        'borderRadius': '12px',
        'fontSize': '12px',
        'fontWeight': '600',
        'marginLeft': '8px'
    },
    'condition_badge_true': {
        'backgroundColor': '#fed7d7',
        'color': '#c53030',
        'padding': '2px 8px',
        'borderRadius': '4px',
        'fontSize': '12px',
        'marginRight': '8px'
    },
    'condition_badge_false': {
        'backgroundColor': '#c6f6d5',
        'color': '#276749',
        'padding': '2px 8px',
        'borderRadius': '4px',
        'fontSize': '12px',
        'marginRight': '8px'
    },
    'feedback_container': {
        'display': 'flex',
        'gap': '12px',
        'marginTop': '16px',
        'paddingTop': '16px',
        'borderTop': '1px solid #e2e8f0'
    },
    'approve_button': {
        'padding': '8px 20px',
        'backgroundColor': '#48bb78',
        'color': 'white',
        'border': 'none',
        'borderRadius': '6px',
        'cursor': 'pointer',
        'fontSize': '14px',
        'fontWeight': '600',
        'display': 'flex',
        'alignItems': 'center',
        'gap': '6px'
    },
    'reject_button': {
        'padding': '8px 20px',
        'backgroundColor': '#fc8181',
        'color': 'white',
        'border': 'none',
        'borderRadius': '6px',
        'cursor': 'pointer',
        'fontSize': '14px',
        'fontWeight': '600',
        'display': 'flex',
        'alignItems': 'center',
        'gap': '6px'
    },
    'feedback_approved': {
        'padding': '8px 16px',
        'backgroundColor': '#c6f6d5',
        'color': '#276749',
        'borderRadius': '6px',
        'fontSize': '14px',
        'fontWeight': '600'
    },
    'feedback_rejected': {
        'padding': '8px 16px',
        'backgroundColor': '#fed7d7',
        'color': '#c53030',
        'borderRadius': '6px',
        'fontSize': '14px',
        'fontWeight': '600'
    },
    'feedback_textarea': {
        'width': '100%',
        'minHeight': '60px',
        'padding': '10px',
        'border': '1px solid #e2e8f0',
        'borderRadius': '6px',
        'fontSize': '14px',
        'resize': 'vertical',
        'marginBottom': '12px',
        'fontFamily': 'inherit'
    },
    'submit_feedback_button': {
        'padding': '8px 16px',
        'backgroundColor': '#3182ce',
        'color': 'white',
        'border': 'none',
        'borderRadius': '6px',
        'cursor': 'pointer',
        'fontSize': '14px',
        'fontWeight': '600'
    },
    'feedback_submitted': {
        'padding': '8px 16px',
        'backgroundColor': '#bee3f8',
        'color': '#2b6cb0',
        'borderRadius': '6px',
        'fontSize': '14px',
        'fontWeight': '600'
    }
}

# Initialize member list on startup
initialize_member_list()

# App layout
app.layout = html.Div([
    # Header
    html.Div([
        html.H1("🏥 Healthcare Offer Recommendations", style=styles['header']),
        html.P("Select a member to view their personalized offer recommendations with AI-generated explanations.",
               style={'textAlign': 'center', 'color': '#718096', 'marginBottom': '30px'})
    ]),
    
    # Member Selection Dropdown
    html.Div([
        html.Label("Select Member:", style={'fontWeight': '600', 'marginBottom': '8px', 'display': 'block'}),
        dcc.Dropdown(
            id='member-dropdown',
            options=[{'label': member_id, 'value': member_id} for member_id in MEMBER_LIST],
            placeholder='Search or select a member...',
            searchable=True,
            clearable=True,
            style={'fontSize': '16px', 'width': '100%', 'minWidth': '400px'}
        ),
        html.Div(
            f"Total members available: {len(MEMBER_LIST):,}",
            style={'marginTop': '8px', 'color': '#718096', 'fontSize': '14px'}
        )
    ], style=styles['search_box']),
    
    # Member Recommendations Display
    dcc.Loading(
        id="loading",
        type="circle",
        color="#3182ce",
        children=html.Div(id='recommendations-container')
    ),
    
    # Store for selected member
    dcc.Store(id='selected-member-store'),
    
    # Store for feedback state
    dcc.Store(id='feedback-store', data={}),
    
    # Hidden div for feedback notifications
    html.Div(id='feedback-notification')
    
], style=styles['container'])

@app.callback(
    Output('recommendations-container', 'children'),
    [Input('member-dropdown', 'value')],
    prevent_initial_call=True
)
def display_member_recommendations(member_id):
    """Display recommendations for selected member."""
    if not member_id:
        return html.Div(
            "👆 Select a member from the dropdown above to view their recommendations.",
            style={'textAlign': 'center', 'color': '#718096', 'padding': '40px', 'fontStyle': 'italic'}
        )
    
    # Fetch recommendations
    recommendations = get_member_recommendations(member_id)
    
    if not recommendations:
        return html.Div(
            f"No recommendations found for member {member_id}.",
            style={'color': '#e53e3e', 'padding': '20px', 'textAlign': 'center'}
        )
    
    # Get member profile from first recommendation
    member = recommendations[0]
    
    # Build member profile card
    profile_section = html.Div([
        html.H2(f"👤 Member: {member_id}", style={'marginBottom': '16px', 'color': '#2d3748'}),
        html.Div([
            create_profile_item("Age", safe_int(member.get('age'), 'N/A')),
            create_profile_item("Risk Score", f"{safe_float(member.get('risk_score')):.1f}"),
            create_profile_item("Chronic Conditions", safe_int(member.get('chronic_condition_count'))),
            create_profile_item("Tenure (months)", safe_int(member.get('tenure_months'), 'N/A')),
            create_profile_item("Total Claims", safe_int(member.get('total_claims'), 'N/A')),
            create_profile_item("Engagements", safe_int(member.get('total_engagements'), 'N/A')),
        ], style=styles['profile_section']),
        
        # Condition flags
        html.Div([
            html.Span("Conditions: ", style={'fontWeight': '600', 'marginRight': '8px'}),
            create_condition_badge("Diabetes", member.get('has_diabetes')),
            create_condition_badge("Cardiovascular", member.get('has_cardiovascular')),
            create_condition_badge("Respiratory", member.get('has_respiratory')),
            create_condition_badge("Mental Health", member.get('has_mental_health')),
            create_condition_badge("Complex Patient", member.get('is_complex_patient')),
        ], style={'marginBottom': '20px'})
    ], style=styles['member_card'])
    
    # Build offer cards
    offer_cards = []
    for rec in recommendations:
        offer_cards.append(create_offer_card(rec, member_id))
    
    return html.Div([
        profile_section,
        html.H3("🎯 Top 5 Recommended Offers", style={'marginBottom': '16px', 'color': '#2d3748'}),
        html.Div(offer_cards)
    ])

def safe_float(value, default=0):
    """Safely convert a value to float."""
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def safe_int(value, default=0):
    """Safely convert a value to int."""
    try:
        return int(float(value)) if value is not None else default
    except (ValueError, TypeError):
        return default

def safe_bool(value):
    """Safely convert a value to bool."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ('true', '1', 'yes')
    try:
        return bool(value)
    except Exception:
        return False

def create_profile_item(label, value):
    """Create a profile stat item."""
    return html.Div([
        html.Div(label, style=styles['profile_label']),
        html.Div(str(value), style=styles['profile_value'])
    ], style=styles['profile_item'])

def create_condition_badge(name, has_condition):
    """Create a condition badge."""
    # Convert to bool safely (database might return string or int)
    has_condition = safe_bool(has_condition)
    style = styles['condition_badge_true'] if has_condition else styles['condition_badge_false']
    icon = "✓" if has_condition else "✗"
    return html.Span(f"{icon} {name}", style=style)

def get_factor_icon(feature_name: str) -> str:
    """Get an appropriate emoji icon for a feature."""
    feature_lower = feature_name.lower()
    
    # Check more specific patterns first (order matters!)
    # Engagement channels - check before generic 'engagement'
    if 'phone' in feature_lower or 'call' in feature_lower:
        return '📞'
    elif 'email' in feature_lower:
        return '📧'
    elif 'app' in feature_lower and 'engagement' in feature_lower:
        return '📲'
    elif 'portal' in feature_lower:
        return '💻'
    elif 'engagement' in feature_lower or 'response' in feature_lower:
        return '📱'
    # Age - check for exact match to avoid 'engagement' false positive
    elif feature_lower == 'age' or feature_lower.startswith('age_') or feature_lower.endswith('_age'):
        return '🎂'
    elif 'risk' in feature_lower:
        return '⚠️'
    elif 'diabetes' in feature_lower:
        return '💉'
    elif 'cardiovascular' in feature_lower or 'heart' in feature_lower:
        return '❤️'
    elif 'respiratory' in feature_lower:
        return '🫁'
    elif 'mental' in feature_lower:
        return '🧠'
    elif 'claim' in feature_lower:
        return '📋'
    elif 'utilization' in feature_lower:
        return '📊'
    elif 'chronic' in feature_lower or 'condition' in feature_lower:
        return '🏥'
    elif 'pharmacy' in feature_lower or 'rx' in feature_lower:
        return '💊'
    elif 'cost' in feature_lower or 'deductible' in feature_lower:
        return '💰'
    elif 'tenure' in feature_lower:
        return '📅'
    elif 'senior' in feature_lower:
        return '👴'
    elif 'complex' in feature_lower:
        return '🔍'
    elif 'visit' in feature_lower or 'er' in feature_lower:
        return '🏨'
    else:
        return '📌'

def format_feature_name(feature_name: str) -> str:
    """Convert feature name to human-readable format."""
    # Remove common prefixes/suffixes and clean up
    name = feature_name.replace('_', ' ').replace('has ', '').replace(' flag', '')
    
    # Specific replacements for clarity (check more specific patterns first)
    replacements = {
        # Engagement channels (check these first - order matters in dict iteration)
        'phone engagement rate': 'Phone outreach response',
        'call engagement rate': 'Phone outreach response',
        'email engagement rate': 'Email engagement',
        'app engagement rate': 'Mobile app engagement',
        'portal login count': 'Portal activity',
        'total engagements': 'Engagement with health programs',
        'days since last engagement': 'Time since last interaction',
        # Response
        'avg response rate': 'Response to outreach',
        # Claims
        'chronic condition count': 'Number of chronic conditions',
        'total claims count': 'Healthcare claims history',
        'claims last': 'Recent claims activity',
        'days since last claim': 'Time since last healthcare visit',
        'er visit count': 'Emergency room visits',
        'inpatient count': 'Hospital stays',
        'specialist visit count': 'Specialist consultations',
        'preventive visit count': 'Preventive care visits',
        # Utilization
        'avg utilization rate': 'Benefits utilization level',
        'pharmacy utilization rate': 'Prescription medication usage',
        'medical utilization rate': 'Medical services usage',
        'preventive utilization rate': 'Preventive care usage',
        'mental health utilization rate': 'Mental health services usage',
        # Cost
        'remaining deductible pct': 'Remaining deductible',
        'remaining oop max pct': 'Out-of-pocket spending room',
        'total member cost': 'Healthcare spending',
        # Profile
        'risk score': 'Health risk assessment',
        'tenure months': 'Membership duration',
        'is senior': 'Senior member status',
        'is complex patient': 'Complex health needs',
        'high risk': 'Elevated health risk',
    }
    
    name_lower = name.lower()
    for key, value in replacements.items():
        if key in name_lower:
            return value
    
    return name.title()

def generate_factor_description(feature_name: str, value, direction: str, member_profile: dict = None) -> str:
    """
    Generate a human-readable description of why a factor matters.
    
    Args:
        feature_name: The name of the feature
        value: The SHAP factor's value (may be encoded/transformed)
        direction: 'increases' or 'decreases'
        member_profile: Optional dict with actual member data (age, risk_score, etc.)
                       Used to get accurate values for display
    """
    feature_lower = feature_name.lower()
    is_positive = direction == 'increases'
    
    # Helper to get actual value from member profile if available
    def get_actual_value(feature_key, default_value):
        """Get actual value from member profile, falling back to SHAP value."""
        if member_profile:
            # Try direct match
            if feature_key in member_profile and member_profile[feature_key] is not None:
                return member_profile[feature_key]
            # Try with underscores
            key_underscore = feature_key.replace(' ', '_')
            if key_underscore in member_profile and member_profile[key_underscore] is not None:
                return member_profile[key_underscore]
        return default_value
    
    # NOTE: Order matters! Check more specific patterns before generic ones.
    # e.g., "engagement" contains "age", so check engagement first.
    
    # Engagement-related factors (check BEFORE 'age' since 'engagement' contains 'age')
    if 'engagement' in feature_lower or 'response' in feature_lower:
        # Determine the type of engagement
        if 'phone' in feature_lower or 'call' in feature_lower:
            channel = "phone"
            channel_desc = "phone outreach"
        elif 'email' in feature_lower:
            channel = "email"
            channel_desc = "email communications"
        elif 'app' in feature_lower or 'portal' in feature_lower:
            channel = "digital"
            channel_desc = "digital channels"
        elif 'response' in feature_lower:
            rate = safe_float(value)
            pct = rate * 100 if rate <= 1 else rate
            if is_positive:
                return f"Their responsiveness to outreach ({pct:.0f}%) suggests they'll engage with this offer."
            else:
                return f"Their response patterns indicate we may need different messaging approaches."
        else:
            channel = "program"
            channel_desc = "health programs"
        
        # Get actual engagement value if available
        actual_engagements = get_actual_value('total_engagements', value)
        rate = safe_float(actual_engagements)
        
        # Check if it's a rate (0-1) or a count
        if rate <= 1 and 'rate' in feature_lower:
            pct = rate * 100
            if is_positive:
                return f"Their {channel} engagement shows strong receptivity to {channel_desc}."
            else:
                return f"Their {channel} engagement suggests trying different communication channels."
        else:
            count = int(rate) if rate > 1 else None
            if is_positive:
                if count:
                    return f"Their engagement history ({count} interactions) indicates openness to health programs."
                else:
                    return f"Their engagement history indicates openness to health programs."
            else:
                return f"Their engagement level suggests exploring alternative outreach methods."
    
    # Age - use word boundary check to avoid matching "engagement"
    elif feature_lower == 'age' or feature_lower.startswith('age_') or feature_lower.endswith('_age'):
        # Get actual age from member profile
        actual_age = get_actual_value('age', value)
        age_val = int(safe_float(actual_age))
        
        # Sanity check - if age seems wrong, don't show specific number
        if age_val < 18 or age_val > 120:
            if is_positive:
                return "Their age demographic makes this offer particularly relevant."
            else:
                return "Based on their age group, other offers may be more relevant."
        
        if is_positive:
            return f"At age {age_val}, this member is in a key demographic for this offer's benefits."
        else:
            return f"At age {age_val}, other offers may be more relevant to their life stage."
    
    # Risk score
    elif 'risk_score' in feature_lower or 'risk score' in feature_lower:
        actual_risk = get_actual_value('risk_score', value)
        val = safe_float(actual_risk)
        if val > 60:
            level = "elevated"
        elif val > 30:
            level = "moderate"
        else:
            level = "lower"
        if is_positive:
            return f"Their {level} risk profile indicates this offer could provide meaningful health support."
        else:
            return f"Their {level} risk profile suggests other programs may be better suited."
    
    # Diabetes
    elif 'diabetes' in feature_lower:
        actual_val = get_actual_value('has_diabetes', value)
        has_condition = safe_bool(actual_val) or safe_float(actual_val) > 0
        if has_condition and is_positive:
            return "Their diabetes diagnosis makes this program particularly relevant for their care needs."
        elif has_condition:
            return "While they have diabetes, other factors suggest different priorities."
        else:
            return "No diabetes diagnosis on record."
    
    # Cardiovascular
    elif 'cardiovascular' in feature_lower or 'heart' in feature_lower:
        actual_val = get_actual_value('has_cardiovascular', value)
        has_condition = safe_bool(actual_val) or safe_float(actual_val) > 0
        if has_condition and is_positive:
            return "Their heart health history makes this program especially beneficial."
        elif has_condition:
            return "Heart health is a consideration, but other factors take priority here."
        else:
            return "No cardiovascular conditions on record."
    
    # Respiratory
    elif 'respiratory' in feature_lower:
        actual_val = get_actual_value('has_respiratory', value)
        has_condition = safe_bool(actual_val) or safe_float(actual_val) > 0
        if has_condition and is_positive:
            return "Their respiratory health needs align well with this offer."
        elif has_condition:
            return "Respiratory health is noted, but other needs may be more pressing."
        else:
            return "No respiratory conditions on record."
    
    # Mental health
    elif 'mental_health' in feature_lower or 'mental health' in feature_lower:
        actual_val = get_actual_value('has_mental_health', value)
        has_condition = safe_bool(actual_val) or safe_float(actual_val) > 0
        if has_condition and is_positive:
            return "Their mental health journey makes this supportive program a great fit."
        elif has_condition:
            return "Mental wellness is important, though other programs may help more."
        else:
            return "No mental health history on record."
    
    # Chronic conditions
    elif 'chronic_condition' in feature_lower or 'chronic condition' in feature_lower:
        actual_count = get_actual_value('chronic_condition_count', value)
        count = int(safe_float(actual_count))
        if count > 2 and is_positive:
            return f"Managing {count} chronic conditions, they can benefit from comprehensive support."
        elif count > 0 and is_positive:
            return f"With {count} chronic condition(s), this targeted support is relevant."
        elif count > 0:
            return f"Their {count} condition(s) are considered, but other factors matter more here."
        else:
            return "No chronic conditions on record."
    
    # Claims
    elif 'claim' in feature_lower:
        actual_claims = get_actual_value('total_claims', value)
        count = int(safe_float(actual_claims))
        if count > 0:
            if is_positive:
                return f"Their healthcare activity ({count} claims) shows engagement with their health."
            else:
                return f"Their claims history ({count}) suggests other approaches may resonate more."
        else:
            if is_positive:
                return "Their healthcare activity shows engagement with their health."
            else:
                return "Their claims history suggests other approaches may resonate more."
    
    # Utilization
    elif 'utilization' in feature_lower:
        actual_util = get_actual_value('avg_utilization_rate', value)
        rate = safe_float(actual_util)
        pct = rate * 100 if rate <= 1 else rate
        if is_positive:
            return f"Their benefits usage shows they actively use their coverage."
        else:
            return f"Their utilization pattern suggests other priorities."
    
    # Pharmacy
    elif 'pharmacy' in feature_lower or 'rx' in feature_lower:
        if is_positive:
            return "Their prescription needs make pharmacy-related benefits valuable."
        else:
            return "Their pharmacy usage suggests other benefits may be more impactful."
    
    # Tenure
    elif 'tenure' in feature_lower:
        actual_tenure = get_actual_value('tenure_months', value)
        months = int(safe_float(actual_tenure))
        years = months // 12
        if months > 0:
            if is_positive:
                if years >= 1:
                    return f"As a member for {years}+ year(s), they've built a relationship with us."
                else:
                    return f"At {months} months, they're getting familiar with available benefits."
            else:
                return "Their membership tenure is a factor in prioritization."
        else:
            if is_positive:
                return "Their membership history shows a relationship with us."
            else:
                return "Their membership tenure is one factor considered."
    
    elif 'senior' in feature_lower:
        actual_senior = get_actual_value('is_senior', value)
        is_senior = safe_bool(actual_senior) or safe_float(actual_senior) > 0
        if is_senior and is_positive:
            return "As a senior member, age-appropriate preventive care is especially valuable."
        elif is_senior:
            return "Senior status is considered, though other factors drive this recommendation."
        else:
            return "Not yet in the senior demographic."
    
    elif 'complex' in feature_lower:
        actual_complex = get_actual_value('is_complex_patient', value)
        is_complex = safe_bool(actual_complex) or safe_float(actual_complex) > 0
        if is_complex and is_positive:
            return "Their complex health needs make coordinated care programs especially helpful."
        elif is_complex:
            return "Complex care needs noted, but other programs may be more targeted."
        else:
            return "Health needs are straightforward at this time."
    
    elif 'deductible' in feature_lower:
        if is_positive:
            return "Their deductible status makes cost-saving programs timely."
        else:
            return "Their deductible status is one factor considered."
    
    # High risk flag
    elif 'high_risk' in feature_lower or 'high risk' in feature_lower:
        actual_risk = get_actual_value('high_risk_flag', value)
        is_high_risk = safe_bool(actual_risk) or safe_float(actual_risk) > 0
        if is_high_risk and is_positive:
            return "Their elevated health risk makes preventive programs especially valuable."
        elif is_high_risk:
            return "Their risk level is considered, but other factors take priority here."
        else:
            return "Their current risk level is within normal range."
    
    else:
        # Generic description - avoid showing potentially incorrect numeric values
        if is_positive:
            return "This factor positively supports this recommendation."
        else:
            return "This factor was considered in the overall assessment."

def create_offer_card(rec, member_id):
    """Create an offer recommendation card with feedback buttons."""
    offer_id = rec.get('offer_id', 'unknown')
    
    # Parse SHAP factors
    shap_factors = []
    try:
        if rec.get('shap_factors'):
            shap_factors = json.loads(rec['shap_factors']) if isinstance(rec['shap_factors'], str) else rec['shap_factors']
    except Exception:
        pass
    
    # Build human-readable SHAP factors display
    # Pass the full rec as member_profile so we can get accurate values
    member_profile = {
        'age': rec.get('age'),
        'risk_score': rec.get('risk_score'),
        'chronic_condition_count': rec.get('chronic_condition_count'),
        'has_diabetes': rec.get('has_diabetes'),
        'has_cardiovascular': rec.get('has_cardiovascular'),
        'has_respiratory': rec.get('has_respiratory'),
        'has_mental_health': rec.get('has_mental_health'),
        'is_complex_patient': rec.get('is_complex_patient'),
        'is_senior': rec.get('is_senior'),
        'tenure_months': rec.get('tenure_months'),
        'total_claims': rec.get('total_claims'),
        'total_engagements': rec.get('total_engagements'),
        'avg_utilization_rate': rec.get('avg_utilization_rate'),
        'high_risk_flag': rec.get('high_risk_flag'),
    }
    
    shap_items = []
    for factor in shap_factors[:5]:  # Top 5 factors
        feature_name = factor.get('feature', '')
        direction = factor.get('direction', 'increases')
        value = factor.get('value')
        is_positive = direction == 'increases'
        
        icon = get_factor_icon(feature_name)
        readable_name = format_feature_name(feature_name)
        description = generate_factor_description(feature_name, value, direction, member_profile)
        
        impact_badge = html.Span(
            "Supports this offer" if is_positive else "Other factors stronger",
            style=styles['impact_badge_positive'] if is_positive else styles['impact_badge_negative']
        )
        
        shap_items.append(
            html.Div([
                html.Span(icon, style=styles['shap_item_icon']),
                html.Div([
                    html.Div([
                        html.Span(readable_name, style=styles['shap_item_title']),
                        impact_badge
                    ]),
                    html.Div(description, style=styles['shap_item_description'])
                ], style=styles['shap_item_content'])
            ], style=styles['shap_item'])
        )
    
    # Create unique IDs for the components
    approve_id = {'type': 'approve-btn', 'member': member_id, 'offer': offer_id}
    reject_id = {'type': 'reject-btn', 'member': member_id, 'offer': offer_id}
    feedback_status_id = {'type': 'feedback-status', 'member': member_id, 'offer': offer_id}
    feedback_text_id = {'type': 'feedback-text', 'member': member_id, 'offer': offer_id}
    submit_text_id = {'type': 'submit-text-btn', 'member': member_id, 'offer': offer_id}
    text_status_id = {'type': 'text-feedback-status', 'member': member_id, 'offer': offer_id}
    explain_btn_id = {'type': 'explain-btn', 'member': member_id, 'offer': offer_id}
    explain_content_id = {'type': 'explain-content', 'member': member_id, 'offer': offer_id}
    
    # Build the explain section (hidden by default)
    explain_section = html.Div([
        html.Div([
            html.Div("🔍 What Influenced This Recommendation", 
                    style={'fontWeight': '600', 'marginBottom': '16px', 'color': '#2d3748', 'fontSize': '16px'}),
            html.P("Based on this member's profile, here's why this offer stands out:",
                  style={'color': '#718096', 'marginBottom': '16px', 'fontSize': '14px'}),
            html.Div(shap_items) if shap_items else html.P(
                "No detailed factor data available for this recommendation.", 
                style={'color': '#718096', 'fontStyle': 'italic'}
            )
        ], style=styles['shap_box'])
    ], id=explain_content_id, style={'display': 'none'}) if shap_factors else None
    
    return html.Div([
        # Header with rank and score
        html.Div([
            html.Div([
                html.Span(f"#{rec.get('rank', 'N/A')}", style=styles['rank_badge']),
                html.H4(rec.get('offer_name', 'Unknown Offer'), 
                       style={'margin': '0 0 0 12px', 'color': '#2d3748'})
            ], style={'display': 'flex', 'alignItems': 'center'}),
            html.Span(f"Score: {safe_float(rec.get('priority_score')):.1f}", style=styles['score_badge'])
        ], style=styles['offer_header']),
        
        # Offer ID
        html.Div(f"Offer ID: {offer_id}", 
                style={'color': '#718096', 'fontSize': '14px', 'marginBottom': '12px'}),
        
        # LLM Reasoning
        html.Div([
            html.Div("💬 Why This Offer?", style={'fontWeight': '600', 'marginBottom': '8px', 'color': '#276749'}),
            html.P(rec.get('llm_reasoning', 'No reasoning available.'), 
                  style={'margin': '0', 'lineHeight': '1.6', 'color': '#2d3748'})
        ], style=styles['reasoning_box']) if rec.get('llm_reasoning') else None,
        
        # Explain Button (only if we have SHAP factors)
        html.Button(
            ["🔍 Explain Key Factors"],
            id=explain_btn_id,
            n_clicks=0,
            style=styles['explain_button']
        ) if shap_factors else None,
        
        # Collapsible Explain Section
        explain_section,
        
        # Feedback buttons
        html.Div([
            html.Div([
                html.Button(
                    ["✓ Approve"],
                    id=approve_id,
                    n_clicks=0,
                    style=styles['approve_button']
                ),
                html.Button(
                    ["✗ Reject"],
                    id=reject_id,
                    n_clicks=0,
                    style=styles['reject_button']
                ),
            ], style={'display': 'flex', 'gap': '12px'}),
            html.Div(id=feedback_status_id, style={'marginLeft': 'auto'})
        ], style=styles['feedback_container']),
        
        # Text feedback section
        html.Div([
            html.Div("💭 Additional Comments", style={'fontWeight': '600', 'marginBottom': '8px', 'color': '#4a5568'}),
            dcc.Textarea(
                id=feedback_text_id,
                placeholder="Share your thoughts on this recommendation (optional)...",
                style=styles['feedback_textarea']
            ),
            html.Div([
                html.Button(
                    "📤 Submit Comment",
                    id=submit_text_id,
                    n_clicks=0,
                    style=styles['submit_feedback_button']
                ),
                html.Div(id=text_status_id, style={'marginLeft': '12px', 'display': 'flex', 'alignItems': 'center'})
            ], style={'display': 'flex', 'alignItems': 'center'})
        ], style={'marginTop': '16px', 'paddingTop': '16px', 'borderTop': '1px dashed #e2e8f0'})
        
    ], style=styles['offer_card'])

@app.callback(
    Output({'type': 'explain-content', 'member': dash.MATCH, 'offer': dash.MATCH}, 'style'),
    Output({'type': 'explain-btn', 'member': dash.MATCH, 'offer': dash.MATCH}, 'children'),
    Output({'type': 'explain-btn', 'member': dash.MATCH, 'offer': dash.MATCH}, 'style'),
    Input({'type': 'explain-btn', 'member': dash.MATCH, 'offer': dash.MATCH}, 'n_clicks'),
    prevent_initial_call=True
)
def toggle_explain_section(n_clicks):
    """Toggle the visibility of the explain section."""
    if n_clicks is None:
        return dash.no_update, dash.no_update, dash.no_update
    
    # Toggle based on odd/even clicks
    is_open = n_clicks % 2 == 1
    
    if is_open:
        return (
            {'display': 'block'},  # Show content
            ["🔍 Hide Key Factors"],  # Update button text
            styles['explain_button_active']  # Active button style
        )
    else:
        return (
            {'display': 'none'},  # Hide content
            ["🔍 Explain Key Factors"],  # Reset button text
            styles['explain_button']  # Default button style
        )

@app.callback(
    Output({'type': 'feedback-status', 'member': dash.MATCH, 'offer': dash.MATCH}, 'children'),
    Output({'type': 'feedback-status', 'member': dash.MATCH, 'offer': dash.MATCH}, 'style'),
    Output({'type': 'approve-btn', 'member': dash.MATCH, 'offer': dash.MATCH}, 'disabled'),
    Output({'type': 'reject-btn', 'member': dash.MATCH, 'offer': dash.MATCH}, 'disabled'),
    [Input({'type': 'approve-btn', 'member': dash.MATCH, 'offer': dash.MATCH}, 'n_clicks'),
     Input({'type': 'reject-btn', 'member': dash.MATCH, 'offer': dash.MATCH}, 'n_clicks')],
    prevent_initial_call=True
)
def handle_feedback(approve_clicks, reject_clicks):
    """Handle approve/reject button clicks."""
    ctx = callback_context
    
    if not ctx.triggered:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Get which button was clicked
    triggered = ctx.triggered[0]
    prop_id = triggered['prop_id']
    
    # Parse the button ID to get member and offer
    button_info = json.loads(prop_id.split('.')[0])
    member_id = button_info['member']
    offer_id = button_info['offer']
    button_type = button_info['type']
    
    if button_type == 'approve-btn' and approve_clicks:
        feedback = 'approved'
        saved = save_feedback(member_id, offer_id, feedback)
        status_text = "✓ Approved" + (" (saved)" if saved else " (not saved)")
        return (
            status_text,
            styles['feedback_approved'],
            True,  # Disable approve button
            True   # Disable reject button
        )
    elif button_type == 'reject-btn' and reject_clicks:
        feedback = 'rejected'
        saved = save_feedback(member_id, offer_id, feedback)
        status_text = "✗ Rejected" + (" (saved)" if saved else " (not saved)")
        return (
            status_text,
            styles['feedback_rejected'],
            True,  # Disable approve button
            True   # Disable reject button
        )
    
    return dash.no_update, dash.no_update, dash.no_update, dash.no_update

@app.callback(
    Output({'type': 'text-feedback-status', 'member': dash.MATCH, 'offer': dash.MATCH}, 'children'),
    Output({'type': 'text-feedback-status', 'member': dash.MATCH, 'offer': dash.MATCH}, 'style'),
    Output({'type': 'submit-text-btn', 'member': dash.MATCH, 'offer': dash.MATCH}, 'disabled'),
    Output({'type': 'feedback-text', 'member': dash.MATCH, 'offer': dash.MATCH}, 'disabled'),
    Input({'type': 'submit-text-btn', 'member': dash.MATCH, 'offer': dash.MATCH}, 'n_clicks'),
    State({'type': 'feedback-text', 'member': dash.MATCH, 'offer': dash.MATCH}, 'value'),
    prevent_initial_call=True
)
def handle_text_feedback(n_clicks, feedback_text):
    """Handle text feedback submission."""
    ctx = callback_context
    
    if not ctx.triggered or not n_clicks:
        return dash.no_update, dash.no_update, dash.no_update, dash.no_update
    
    # Get the button ID to extract member and offer
    triggered = ctx.triggered[0]
    prop_id = triggered['prop_id']
    button_info = json.loads(prop_id.split('.')[0])
    member_id = button_info['member']
    offer_id = button_info['offer']
    
    if not feedback_text or not feedback_text.strip():
        return (
            "⚠️ Please enter a comment",
            {'color': '#d69e2e', 'fontSize': '14px'},
            False,
            False
        )
    
    # Save the text feedback
    saved = save_feedback(member_id, offer_id, 'comment', feedback_text.strip())
    status_text = "✓ Comment submitted!" if saved else "⚠️ Comment logged (DB save failed)"
    status_style = styles['feedback_submitted'] if saved else {'color': '#d69e2e', 'fontSize': '14px'}
    
    return (
        status_text,
        status_style,
        True,   # Disable submit button
        True    # Disable textarea
    )

if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8050)
