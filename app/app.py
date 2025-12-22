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
        'backgroundColor': '#ebf8ff',
        'border': '1px solid #90cdf4',
        'borderRadius': '8px',
        'padding': '16px',
        'marginTop': '12px'
    },
    'shap_item': {
        'display': 'flex',
        'justifyContent': 'space-between',
        'padding': '8px 0',
        'borderBottom': '1px solid #e2e8f0'
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
    
    # Build SHAP factors display
    shap_items = []
    for factor in shap_factors[:5]:  # Top 5 factors
        direction_icon = "↑" if factor.get('direction') == 'increases' else "↓"
        direction_color = "#38a169" if factor.get('direction') == 'increases' else "#e53e3e"
        
        shap_items.append(
            html.Div([
                html.Span(factor.get('feature', '').replace('_', ' ').title(), 
                         style={'fontWeight': '500'}),
                html.Span([
                    html.Span(f"Value: {safe_float(factor.get('value'), 'N/A')} ", 
                             style={'color': '#718096', 'marginRight': '10px'}),
                    html.Span(f"{direction_icon} {abs(safe_float(factor.get('shap_value'))):.3f}", 
                             style={'color': direction_color, 'fontWeight': '600'})
                ])
            ], style=styles['shap_item'])
        )
    
    # Create unique IDs for the feedback components
    approve_id = {'type': 'approve-btn', 'member': member_id, 'offer': offer_id}
    reject_id = {'type': 'reject-btn', 'member': member_id, 'offer': offer_id}
    feedback_status_id = {'type': 'feedback-status', 'member': member_id, 'offer': offer_id}
    feedback_text_id = {'type': 'feedback-text', 'member': member_id, 'offer': offer_id}
    submit_text_id = {'type': 'submit-text-btn', 'member': member_id, 'offer': offer_id}
    text_status_id = {'type': 'text-feedback-status', 'member': member_id, 'offer': offer_id}
    
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
        
        # SHAP Factors
        html.Div([
            html.Div("📊 Key Factors", style={'fontWeight': '600', 'marginBottom': '12px', 'color': '#2b6cb0'}),
            html.Div(shap_items) if shap_items else html.P("No factor data available.", 
                                                           style={'color': '#718096', 'fontStyle': 'italic'})
        ], style=styles['shap_box']) if shap_factors else None,
        
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
