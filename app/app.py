"""
Offer Management Console - Lakebase Version

This app uses Databricks Lakebase (PostgreSQL-compatible) for fast data retrieval.
Uses the Databricks App's service principal (DATABRICKS_CLIENT_ID/SECRET) to 
generate OAuth tokens for Lakebase authentication.

Required App Resource:
- Type: Database
- Database: demos
- Instance: offer-prioritization
"""

import dash
from dash import html, dcc, Input, Output, State, callback_context
import os
import json
import time
import urllib.request
import urllib.parse

# PostgreSQL driver for Lakebase connection
import psycopg2
from psycopg2.extras import RealDictCursor

# Configuration
CATALOG = "demos"
SCHEMA = "offer_prioritization"
LAKEBASE_TABLE = "lakebase_offers"

# Delta table for feedback (NOT in Lakebase - written via Databricks SQL API)
FEEDBACK_TABLE_DELTA = f"{CATALOG}.{SCHEMA}.offer_feedback"

# Databricks App service principal credentials (auto-injected)
DATABRICKS_CLIENT_ID = os.getenv("DATABRICKS_CLIENT_ID", "")
DATABRICKS_CLIENT_SECRET = os.getenv("DATABRICKS_CLIENT_SECRET", "")
DATABRICKS_HOST = os.getenv("DATABRICKS_HOST", "")  # e.g., fe-vm-vdm-classic-h4yue1.cloud.databricks.com
DATABRICKS_WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")

# PostgreSQL/Lakebase connection settings
PG_HOST = os.getenv("PGHOST", "")
PG_PORT = os.getenv("PGPORT", "5432")
PG_DATABASE = os.getenv("PGDATABASE", CATALOG)
PG_USER = os.getenv("PGUSER", "")  # Service principal ID (will use CLIENT_ID if not set)
PG_SSLMODE = os.getenv("PGSSLMODE", "require")

# Full table path in Lakebase: schema.table (within the PGDATABASE)
FULL_TABLE_NAME = f"{SCHEMA}.{LAKEBASE_TABLE}"

# Token cache
_token_cache = {
    "token": None,
    "expires_at": 0
}


def get_workspace_host():
    """Extract the workspace host from DATABRICKS_HOST."""
    host = DATABRICKS_HOST
    # Remove any protocol prefix
    if host.startswith("https://"):
        host = host[8:]
    if host.startswith("http://"):
        host = host[7:]
    # Handle fe-vm-vdm-classic-xxx.cloud.databricks.com format
    # Extract just the workspace part (e.g., h4yue1.cloud.databricks.com)
    if "cloud.databricks.com" in host:
        parts = host.split(".")
        # Find the part before 'cloud'
        for i, part in enumerate(parts):
            if part == "cloud":
                # Take from here to end
                return ".".join(parts[i-1:])
    return host


def get_oauth_token():
    """
    Get an OAuth token for the service principal using client credentials flow.
    Caches the token and refreshes when expired.
    """
    global _token_cache
    
    # Check if we have a valid cached token (with 60s buffer)
    if _token_cache["token"] and time.time() < _token_cache["expires_at"] - 60:
        return _token_cache["token"]
    
    if not DATABRICKS_CLIENT_ID or not DATABRICKS_CLIENT_SECRET:
        print("⚠️ DATABRICKS_CLIENT_ID or DATABRICKS_CLIENT_SECRET not set")
        # Fall back to PGPASSWORD or LAKEBASE_PASSWORD if available
        fallback = os.getenv("PGPASSWORD", "") or os.getenv("LAKEBASE_PASSWORD", "")
        if fallback:
            print("📌 Using fallback password from environment")
            return fallback
        raise Exception(
            "No authentication credentials available. "
            "Make sure DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET are set."
        )
    
    # Get workspace host for token endpoint
    workspace_host = get_workspace_host()
    if not workspace_host:
        raise Exception("Cannot determine workspace host from DATABRICKS_HOST")
    
    token_url = f"https://{workspace_host}/oidc/v1/token"
    
    print(f"🔑 Requesting OAuth token from {token_url}")
    
    # Prepare the request
    data = urllib.parse.urlencode({
        "grant_type": "client_credentials",
        "scope": "all-apis"
    }).encode("utf-8")
    
    # Create request with Basic auth
    import base64
    credentials = base64.b64encode(
        f"{DATABRICKS_CLIENT_ID}:{DATABRICKS_CLIENT_SECRET}".encode()
    ).decode()
    
    req = urllib.request.Request(
        token_url,
        data=data,
        headers={
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {credentials}"
        }
    )
    
    try:
        with urllib.request.urlopen(req, timeout=30) as response:
            result = json.loads(response.read().decode())
            
            access_token = result.get("access_token")
            expires_in = result.get("expires_in", 3600)  # Default 1 hour
            
            # Cache the token
            _token_cache["token"] = access_token
            _token_cache["expires_at"] = time.time() + expires_in
            
            print(f"✅ OAuth token obtained (expires in {expires_in}s)")
            return access_token
            
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        print(f"❌ OAuth token request failed: {e.code} - {error_body}")
        raise Exception(f"Failed to get OAuth token: {e.code} - {error_body}")
    except Exception as e:
        print(f"❌ OAuth token request error: {e}")
        raise


# Print configuration at startup
print(f"🔗 Lakebase Connection Configuration:")
print(f"   Host: {PG_HOST}")
print(f"   Port: {PG_PORT}")
print(f"   Database: {PG_DATABASE}")
print(f"   User: {PG_USER[:30]}..." if PG_USER else "   User: NOT SET")
print(f"   Table: {FULL_TABLE_NAME}")
print(f"")
print(f"🔐 Service Principal Auth:")
print(f"   DATABRICKS_CLIENT_ID: {'SET' if DATABRICKS_CLIENT_ID else 'NOT SET'}")
print(f"   DATABRICKS_CLIENT_SECRET: {'SET' if DATABRICKS_CLIENT_SECRET else 'NOT SET'}")
print(f"   DATABRICKS_HOST: {DATABRICKS_HOST}")
print(f"   DATABRICKS_WAREHOUSE_ID: {'SET' if DATABRICKS_WAREHOUSE_ID else 'NOT SET'}")
print(f"")
print(f"📝 Feedback Delta Table: {FEEDBACK_TABLE_DELTA}")


def execute_databricks_sql(sql: str, wait_timeout: str = "30s"):
    """
    Execute SQL against Databricks using the SQL Statement Execution API.
    This is used to write to Delta tables (not Lakebase).
    
    Args:
        sql: The SQL statement to execute
        wait_timeout: How long to wait for the statement to complete
    
    Returns:
        True if successful, False otherwise
    """
    if not DATABRICKS_WAREHOUSE_ID:
        print("⚠️ DATABRICKS_WAREHOUSE_ID not set - cannot execute Databricks SQL")
        return False
    
    workspace_host = get_workspace_host()
    if not workspace_host:
        print("⚠️ Cannot determine workspace host")
        return False
    
    api_url = f"https://{workspace_host}/api/2.0/sql/statements"
    
    # Get OAuth token
    token = get_oauth_token()
    
    # Prepare request
    payload = json.dumps({
        "warehouse_id": DATABRICKS_WAREHOUSE_ID,
        "statement": sql,
        "wait_timeout": wait_timeout
    }).encode("utf-8")
    
    req = urllib.request.Request(
        api_url,
        data=payload,
        headers={
            "Content-Type": "application/json",
            "Authorization": f"Bearer {token}"
        }
    )
    
    try:
        with urllib.request.urlopen(req, timeout=60) as response:
            result = json.loads(response.read().decode())
            status = result.get("status", {}).get("state", "")
            
            if status == "SUCCEEDED":
                return True
            elif status == "FAILED":
                error = result.get("status", {}).get("error", {})
                print(f"❌ Databricks SQL failed: {error.get('message', 'Unknown error')}")
                return False
            else:
                # Statement is still running or pending
                print(f"⏳ Databricks SQL status: {status}")
                return True  # Assume it will complete
                
    except urllib.error.HTTPError as e:
        error_body = e.read().decode() if e.fp else ""
        print(f"❌ Databricks SQL API error: {e.code} - {error_body}")
        return False
    except Exception as e:
        print(f"❌ Databricks SQL error: {e}")
        return False


def get_connection():
    """
    Create a connection to Lakebase using the service principal's OAuth token.
    The token is obtained using DATABRICKS_CLIENT_ID and DATABRICKS_CLIENT_SECRET.
    """
    if not PG_HOST:
        raise Exception(
            "PGHOST not configured. "
            "Make sure your Databricks App has a Database resource linked to your Lakebase instance."
        )
    
    # Use PGUSER if set, otherwise use the service principal client ID
    user = PG_USER or DATABRICKS_CLIENT_ID
    if not user:
        raise Exception(
            "No user configured. Set PGUSER or DATABRICKS_CLIENT_ID."
        )
    
    # Get OAuth token (cached and auto-refreshed)
    password = get_oauth_token()
    
    print(f"🔌 Connecting to Lakebase as {user[:30]}...")
    
    return psycopg2.connect(
        host=PG_HOST,
        port=int(PG_PORT),
        database=PG_DATABASE,
        user=user,
        password=password,
        sslmode=PG_SSLMODE
    )


def execute_query(query: str):
    """
    Execute a SQL query against Lakebase.
    Returns list of dictionaries.
    """
    try:
        conn = get_connection()
        with conn.cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query)
            results = cursor.fetchall()
        conn.close()
        return [dict(row) for row in results]
    except Exception as e:
        print(f"❌ Query error: {e}")
        print(f"   Query: {query[:200]}...")
        return []


def execute_write(query: str):
    """
    Execute a write query (INSERT/UPDATE/DELETE) against Lakebase.
    """
    try:
        conn = get_connection()
        with conn.cursor() as cursor:
            cursor.execute(query)
        conn.commit()
        conn.close()
        return True
    except Exception as e:
        print(f"❌ Write error: {e}")
        return False


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
    """Create the feedback Delta table if it doesn't exist."""
    global FEEDBACK_TABLE_CREATED
    if FEEDBACK_TABLE_CREATED:
        return True
    
    try:
        # Create Delta table if it doesn't exist (using Databricks SQL API)
        create_query = f"""
            CREATE TABLE IF NOT EXISTS {FEEDBACK_TABLE_DELTA} (
                member_id STRING,
                offer_id STRING,
                feedback STRING,
                feedback_text STRING,
                feedback_time TIMESTAMP
            )
            USING DELTA
        """
        if execute_databricks_sql(create_query):
            FEEDBACK_TABLE_CREATED = True
            print(f"✅ Feedback table created: {FEEDBACK_TABLE_DELTA}")
            return True
        return False
    except Exception as e:
        print(f"⚠️ Could not create feedback table: {e}")
        return False


def save_feedback(member_id: str, offer_id: str, feedback: str, feedback_text: str = None):
    """Save offer feedback to Delta table using Databricks SQL API."""
    text_display = f" - Comment: {feedback_text[:50]}..." if feedback_text and len(feedback_text) > 50 else (f" - Comment: {feedback_text}" if feedback_text else "")
    print(f"📝 Feedback received: {member_id} - {offer_id} - {feedback}{text_display}")
    
    ensure_feedback_table()
    
    try:
        safe_text = feedback_text.replace("'", "''") if feedback_text else ""
        query = f"""
            INSERT INTO {FEEDBACK_TABLE_DELTA} (member_id, offer_id, feedback, feedback_text, feedback_time) 
            VALUES ('{member_id}', '{offer_id}', '{feedback}', '{safe_text}', current_timestamp())
        """
        if execute_databricks_sql(query):
            print(f"✅ Saved to Delta table: {FEEDBACK_TABLE_DELTA}")
            return True
        return False
    except Exception as e:
        print(f"⚠️ Could not save feedback to Delta table: {e}")
        return False


# Cache member list on startup
MEMBER_LIST = []
INIT_ERROR = None

def initialize_member_list():
    """Initialize the member list cache."""
    global MEMBER_LIST, INIT_ERROR
    try:
        print(f"🔄 Loading members from {FULL_TABLE_NAME}...")
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

# Custom CSS styles - Business/Marketing Professional Theme
styles = {
    # Page wrapper
    'page_wrapper': {
        'minHeight': '100vh',
        'backgroundColor': '#f4f6f9',
        'fontFamily': '"Inter", -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif'
    },
    # Top navigation bar
    'navbar': {
        'backgroundColor': '#0f172a',
        'padding': '0 40px',
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'space-between',
        'height': '64px',
        'boxShadow': '0 1px 3px rgba(0,0,0,0.12)',
        'position': 'sticky',
        'top': '0',
        'zIndex': '100'
    },
    'navbar_brand': {
        'display': 'flex',
        'alignItems': 'center',
        'gap': '12px'
    },
    'navbar_logo': {
        'fontSize': '24px'
    },
    'navbar_title': {
        'color': 'white',
        'fontSize': '18px',
        'fontWeight': '600',
        'margin': '0',
        'letterSpacing': '-0.5px'
    },
    'navbar_subtitle': {
        'color': '#94a3b8',
        'fontSize': '12px',
        'margin': '0',
        'fontWeight': '400'
    },
    'navbar_status': {
        'display': 'flex',
        'alignItems': 'center',
        'gap': '8px',
        'color': '#94a3b8',
        'fontSize': '13px'
    },
    'status_dot': {
        'width': '8px',
        'height': '8px',
        'backgroundColor': '#22c55e',
        'borderRadius': '50%',
        'display': 'inline-block'
    },
    # Main container
    'container': {
        'maxWidth': '1400px',
        'margin': '0 auto',
        'padding': '32px 40px',
    },
    'header': {
        'textAlign': 'center',
        'marginBottom': '30px',
        'color': '#1a365d'
    },
    # Search/filter section
    'search_box': {
        'display': 'block',
        'marginBottom': '32px',
        'padding': '24px 28px',
        'backgroundColor': 'white',
        'borderRadius': '8px',
        'boxShadow': '0 1px 3px rgba(0,0,0,0.08)',
        'border': '1px solid #e2e8f0'
    },
    'search_header': {
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'space-between',
        'marginBottom': '16px'
    },
    'section_title': {
        'fontSize': '14px',
        'fontWeight': '600',
        'color': '#374151',
        'textTransform': 'uppercase',
        'letterSpacing': '0.5px',
        'margin': '0'
    },
    'input': {
        'flex': '1',
        'padding': '12px 16px',
        'fontSize': '15px',
        'border': '1px solid #d1d5db',
        'borderRadius': '6px',
        'outline': 'none'
    },
    'button': {
        'padding': '10px 20px',
        'backgroundColor': '#1e40af',
        'color': 'white',
        'border': 'none',
        'borderRadius': '6px',
        'cursor': 'pointer',
        'fontSize': '14px',
        'fontWeight': '500'
    },
    # Member profile card
    'member_card': {
        'backgroundColor': 'white',
        'borderRadius': '8px',
        'padding': '24px 28px',
        'marginBottom': '24px',
        'boxShadow': '0 1px 3px rgba(0,0,0,0.08)',
        'border': '1px solid #e2e8f0'
    },
    'member_header': {
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'space-between',
        'marginBottom': '20px',
        'paddingBottom': '16px',
        'borderBottom': '1px solid #e5e7eb'
    },
    'member_title': {
        'fontSize': '18px',
        'fontWeight': '600',
        'color': '#111827',
        'margin': '0'
    },
    'member_id_badge': {
        'backgroundColor': '#f3f4f6',
        'color': '#6b7280',
        'padding': '4px 12px',
        'borderRadius': '4px',
        'fontSize': '13px',
        'fontFamily': 'monospace'
    },
    'profile_section': {
        'display': 'grid',
        'gridTemplateColumns': 'repeat(auto-fit, minmax(140px, 1fr))',
        'gap': '20px',
        'marginBottom': '20px',
        'padding': '20px',
        'backgroundColor': '#f9fafb',
        'borderRadius': '6px',
        'border': '1px solid #f3f4f6'
    },
    'profile_item': {
        'textAlign': 'center'
    },
    'profile_label': {
        'fontSize': '11px',
        'color': '#6b7280',
        'textTransform': 'uppercase',
        'letterSpacing': '0.5px',
        'marginBottom': '6px',
        'fontWeight': '500'
    },
    'profile_value': {
        'fontSize': '22px',
        'fontWeight': '700',
        'color': '#111827'
    },
    # Offer cards
    'offers_section_header': {
        'display': 'flex',
        'alignItems': 'center',
        'justifyContent': 'space-between',
        'marginBottom': '16px'
    },
    'offers_title': {
        'fontSize': '16px',
        'fontWeight': '600',
        'color': '#111827',
        'margin': '0'
    },
    'offer_card': {
        'backgroundColor': 'white',
        'border': '1px solid #e5e7eb',
        'borderRadius': '8px',
        'padding': '20px 24px',
        'marginBottom': '12px',
        'transition': 'border-color 0.2s ease',
        'boxShadow': '0 1px 2px rgba(0,0,0,0.04)'
    },
    'offer_header': {
        'display': 'flex',
        'justifyContent': 'space-between',
        'alignItems': 'flex-start',
        'marginBottom': '12px'
    },
    'offer_title_section': {
        'display': 'flex',
        'alignItems': 'center',
        'gap': '12px'
    },
    'rank_badge': {
        'backgroundColor': '#1e40af',
        'color': 'white',
        'padding': '4px 10px',
        'borderRadius': '4px',
        'fontSize': '12px',
        'fontWeight': '600',
        'minWidth': '28px',
        'textAlign': 'center'
    },
    'offer_name': {
        'fontSize': '16px',
        'fontWeight': '600',
        'color': '#111827',
        'margin': '0'
    },
    'offer_id_text': {
        'fontSize': '12px',
        'color': '#9ca3af',
        'marginTop': '2px'
    },
    'score_badge': {
        'backgroundColor': '#dcfce7',
        'color': '#166534',
        'padding': '6px 14px',
        'borderRadius': '4px',
        'fontSize': '14px',
        'fontWeight': '600'
    },
    'score_label': {
        'fontSize': '10px',
        'color': '#166534',
        'textTransform': 'uppercase',
        'letterSpacing': '0.5px',
        'display': 'block',
        'marginBottom': '2px'
    },
    # Reasoning box
    'reasoning_box': {
        'backgroundColor': '#f0fdf4',
        'border': '1px solid #bbf7d0',
        'borderLeft': '3px solid #22c55e',
        'borderRadius': '4px',
        'padding': '16px 20px',
        'marginTop': '16px',
        'marginBottom': '16px'
    },
    'reasoning_label': {
        'fontSize': '11px',
        'fontWeight': '600',
        'color': '#166534',
        'textTransform': 'uppercase',
        'letterSpacing': '0.5px',
        'marginBottom': '8px'
    },
    'reasoning_text': {
        'margin': '0',
        'lineHeight': '1.6',
        'color': '#374151',
        'fontSize': '14px'
    },
    # Explain/Factors section
    'shap_box': {
        'backgroundColor': '#fafafa',
        'border': '1px solid #e5e7eb',
        'borderRadius': '6px',
        'padding': '20px 24px',
        'marginTop': '16px'
    },
    'shap_header': {
        'fontSize': '13px',
        'fontWeight': '600',
        'color': '#374151',
        'marginBottom': '16px',
        'display': 'flex',
        'alignItems': 'center',
        'gap': '8px'
    },
    'shap_item': {
        'display': 'flex',
        'alignItems': 'flex-start',
        'padding': '14px 0',
        'borderBottom': '1px solid #f3f4f6',
        'gap': '14px'
    },
    'shap_item_icon': {
        'fontSize': '18px',
        'lineHeight': '1',
        'marginTop': '2px',
        'width': '24px',
        'textAlign': 'center'
    },
    'shap_item_content': {
        'flex': '1'
    },
    'shap_item_title': {
        'fontWeight': '600',
        'color': '#1f2937',
        'marginBottom': '4px',
        'fontSize': '14px',
        'display': 'flex',
        'alignItems': 'center',
        'flexWrap': 'wrap',
        'gap': '8px'
    },
    'shap_item_description': {
        'color': '#6b7280',
        'fontSize': '13px',
        'lineHeight': '1.5'
    },
    'explain_button': {
        'padding': '8px 16px',
        'backgroundColor': 'white',
        'color': '#374151',
        'border': '1px solid #d1d5db',
        'borderRadius': '6px',
        'cursor': 'pointer',
        'fontSize': '13px',
        'fontWeight': '500',
        'display': 'inline-flex',
        'alignItems': 'center',
        'gap': '6px',
        'marginTop': '12px',
        'transition': 'all 0.15s ease'
    },
    'explain_button_active': {
        'padding': '8px 16px',
        'backgroundColor': '#1e40af',
        'color': 'white',
        'border': '1px solid #1e40af',
        'borderRadius': '6px',
        'cursor': 'pointer',
        'fontSize': '13px',
        'fontWeight': '500',
        'display': 'inline-flex',
        'alignItems': 'center',
        'gap': '6px',
        'marginTop': '12px'
    },
    'impact_badge_positive': {
        'display': 'inline-block',
        'backgroundColor': '#dcfce7',
        'color': '#166534',
        'padding': '2px 8px',
        'borderRadius': '3px',
        'fontSize': '11px',
        'fontWeight': '500'
    },
    'impact_badge_negative': {
        'display': 'inline-block',
        'backgroundColor': '#fef2f2',
        'color': '#991b1b',
        'padding': '2px 8px',
        'borderRadius': '3px',
        'fontSize': '11px',
        'fontWeight': '500'
    },
    'condition_badge_true': {
        'backgroundColor': '#fef2f2',
        'color': '#991b1b',
        'padding': '3px 10px',
        'borderRadius': '4px',
        'fontSize': '12px',
        'marginRight': '8px',
        'fontWeight': '500'
    },
    'condition_badge_false': {
        'backgroundColor': '#f0fdf4',
        'color': '#166534',
        'padding': '3px 10px',
        'borderRadius': '4px',
        'fontSize': '12px',
        'marginRight': '8px',
        'fontWeight': '500'
    },
    # Feedback section
    'feedback_section': {
        'marginTop': '20px',
        'paddingTop': '20px',
        'borderTop': '1px solid #e5e7eb'
    },
    'feedback_label': {
        'fontSize': '12px',
        'fontWeight': '600',
        'color': '#6b7280',
        'textTransform': 'uppercase',
        'letterSpacing': '0.5px',
        'marginBottom': '12px'
    },
    'feedback_container': {
        'display': 'flex',
        'gap': '10px',
        'alignItems': 'center',
        'flexWrap': 'wrap'
    },
    'approve_button': {
        'padding': '8px 16px',
        'backgroundColor': '#16a34a',
        'color': 'white',
        'border': 'none',
        'borderRadius': '5px',
        'cursor': 'pointer',
        'fontSize': '13px',
        'fontWeight': '500',
        'display': 'inline-flex',
        'alignItems': 'center',
        'gap': '6px',
        'transition': 'background-color 0.15s ease'
    },
    'reject_button': {
        'padding': '8px 16px',
        'backgroundColor': '#dc2626',
        'color': 'white',
        'border': 'none',
        'borderRadius': '5px',
        'cursor': 'pointer',
        'fontSize': '13px',
        'fontWeight': '500',
        'display': 'inline-flex',
        'alignItems': 'center',
        'gap': '6px',
        'transition': 'background-color 0.15s ease'
    },
    'feedback_approved': {
        'padding': '8px 14px',
        'backgroundColor': '#dcfce7',
        'color': '#166534',
        'borderRadius': '5px',
        'fontSize': '13px',
        'fontWeight': '600',
        'display': 'inline-flex',
        'alignItems': 'center',
        'gap': '6px'
    },
    'feedback_rejected': {
        'padding': '8px 14px',
        'backgroundColor': '#fef2f2',
        'color': '#991b1b',
        'borderRadius': '5px',
        'fontSize': '13px',
        'fontWeight': '600',
        'display': 'inline-flex',
        'alignItems': 'center',
        'gap': '6px'
    },
    # Comment section
    'comment_section': {
        'marginTop': '16px',
        'padding': '16px',
        'backgroundColor': '#f9fafb',
        'borderRadius': '6px',
        'border': '1px solid #f3f4f6'
    },
    'feedback_textarea': {
        'width': '100%',
        'minHeight': '70px',
        'padding': '12px',
        'border': '1px solid #d1d5db',
        'borderRadius': '5px',
        'fontSize': '13px',
        'resize': 'vertical',
        'marginBottom': '12px',
        'fontFamily': 'inherit',
        'backgroundColor': 'white'
    },
    'submit_feedback_button': {
        'padding': '8px 16px',
        'backgroundColor': '#1e40af',
        'color': 'white',
        'border': 'none',
        'borderRadius': '5px',
        'cursor': 'pointer',
        'fontSize': '13px',
        'fontWeight': '500'
    },
    'feedback_submitted': {
        'padding': '8px 14px',
        'backgroundColor': '#dbeafe',
        'color': '#1e40af',
        'borderRadius': '5px',
        'fontSize': '13px',
        'fontWeight': '500'
    },
    # Empty state
    'empty_state': {
        'textAlign': 'center',
        'padding': '60px 40px',
        'color': '#6b7280'
    },
    'empty_state_icon': {
        'fontSize': '48px',
        'marginBottom': '16px',
        'opacity': '0.5'
    },
    'empty_state_text': {
        'fontSize': '15px',
        'margin': '0'
    }
}

# Initialize member list on startup
initialize_member_list()

# App layout
app.layout = html.Div([
    # Top Navigation Bar
    html.Div([
        # Brand/Logo section
        html.Div([
            html.Span("◈", style=styles['navbar_logo']),
            html.Div([
                html.H1("Offer Management Console", style=styles['navbar_title']),
                html.P("Powered by Lakebase", style=styles['navbar_subtitle'])
            ])
        ], style=styles['navbar_brand']),
        
        # Status indicator
        html.Div([
            html.Span(style=styles['status_dot']),
            html.Span(f"{len(MEMBER_LIST):,} members • Lakebase")
        ], style=styles['navbar_status'])
    ], style=styles['navbar']),
    
    # Main Content Container
    html.Div([
        # Member Selection Section
        html.Div([
            html.Div([
                html.H3("Member Lookup", style=styles['section_title']),
                html.Span(
                    "Search by ID or browse all members",
                    style={'fontSize': '13px', 'color': '#9ca3af'}
                )
            ], style=styles['search_header']),
            
            dcc.Dropdown(
                id='member-dropdown',
                options=[{'label': f"{member_id}", 'value': member_id} for member_id in MEMBER_LIST],
                placeholder='Enter member ID or select from list...',
                searchable=True,
                clearable=True,
                style={'fontSize': '14px'}
            ),
        ], style=styles['search_box']),
        
        # Recommendations Display Area
        dcc.Loading(
            id="loading",
            type="default",
            color="#1e40af",
            children=html.Div(id='recommendations-container')
        ),
        
    ], style=styles['container']),
    
    # Store for selected member
    dcc.Store(id='selected-member-store'),
    
    # Store for feedback state
    dcc.Store(id='feedback-store', data={}),
    
    # Hidden div for feedback notifications
    html.Div(id='feedback-notification')
    
], style=styles['page_wrapper'])


@app.callback(
    Output('recommendations-container', 'children'),
    [Input('member-dropdown', 'value')],
    prevent_initial_call=True
)
def display_member_recommendations(member_id):
    """Display recommendations for selected member."""
    if not member_id:
        return html.Div([
            html.Div("📋", style=styles['empty_state_icon']),
            html.P("Select a member from the dropdown above to view their personalized offer recommendations.", 
                   style=styles['empty_state_text'])
        ], style=styles['empty_state'])
    
    # Fetch recommendations
    recommendations = get_member_recommendations(member_id)
    
    if not recommendations:
        return html.Div([
            html.Div("⚠️", style=styles['empty_state_icon']),
            html.P(f"No recommendations found for member {member_id}.", style=styles['empty_state_text'])
        ], style=styles['empty_state'])
    
    # Get member profile from first recommendation
    member = recommendations[0]
    
    # Build member profile card
    profile_section = html.Div([
        # Header with member ID
        html.Div([
            html.H2("Member Profile", style=styles['member_title']),
            html.Span(member_id, style=styles['member_id_badge'])
        ], style=styles['member_header']),
        
        # Key metrics grid
        html.Div([
            create_profile_item("Age", safe_int(member.get('age'), 'N/A')),
            create_profile_item("Risk Score", f"{safe_float(member.get('risk_score')):.0f}"),
            create_profile_item("Conditions", safe_int(member.get('chronic_condition_count'))),
            create_profile_item("Tenure", f"{safe_int(member.get('tenure_months'), 0)} mo"),
            create_profile_item("Claims", safe_int(member.get('total_claims'), 'N/A')),
            create_profile_item("Engagements", safe_int(member.get('total_engagements'), 'N/A')),
        ], style=styles['profile_section']),
        
        # Health indicators
        html.Div([
            html.Span("Health Indicators: ", style={'fontWeight': '500', 'marginRight': '10px', 'color': '#374151', 'fontSize': '13px'}),
            create_condition_badge("Diabetes", member.get('has_diabetes')),
            create_condition_badge("Cardiovascular", member.get('has_cardiovascular')),
            create_condition_badge("Respiratory", member.get('has_respiratory')),
            create_condition_badge("Mental Health", member.get('has_mental_health')),
            create_condition_badge("Complex Care", member.get('is_complex_patient')),
        ], style={'display': 'flex', 'alignItems': 'center', 'flexWrap': 'wrap', 'gap': '4px'})
    ], style=styles['member_card'])
    
    # Build offer cards
    offer_cards = []
    for rec in recommendations:
        offer_cards.append(create_offer_card(rec, member_id))
    
    return html.Div([
        profile_section,
        # Offers section header
        html.Div([
            html.H3("Recommended Offers", style=styles['offers_title']),
            html.Span(f"{len(recommendations)} offers ranked by relevance", 
                     style={'fontSize': '13px', 'color': '#6b7280'})
        ], style=styles['offers_section_header']),
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
    has_condition = safe_bool(has_condition)
    style = styles['condition_badge_true'] if has_condition else styles['condition_badge_false']
    icon = "✓" if has_condition else "✗"
    return html.Span(f"{icon} {name}", style=style)


def get_factor_icon(feature_name: str) -> str:
    """Get an appropriate emoji icon for a feature."""
    feature_lower = feature_name.lower()
    
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
    name = feature_name.replace('_', ' ').replace('has ', '').replace(' flag', '')
    
    replacements = {
        'phone engagement rate': 'Phone outreach response',
        'call engagement rate': 'Phone outreach response',
        'email engagement rate': 'Email engagement',
        'app engagement rate': 'Mobile app engagement',
        'portal login count': 'Portal activity',
        'total engagements': 'Engagement with health programs',
        'days since last engagement': 'Time since last interaction',
        'avg response rate': 'Response to outreach',
        'chronic condition count': 'Number of chronic conditions',
        'total claims count': 'Healthcare claims history',
        'claims last': 'Recent claims activity',
        'days since last claim': 'Time since last healthcare visit',
        'er visit count': 'Emergency room visits',
        'inpatient count': 'Hospital stays',
        'specialist visit count': 'Specialist consultations',
        'preventive visit count': 'Preventive care visits',
        'avg utilization rate': 'Benefits utilization level',
        'pharmacy utilization rate': 'Prescription medication usage',
        'medical utilization rate': 'Medical services usage',
        'preventive utilization rate': 'Preventive care usage',
        'mental health utilization rate': 'Mental health services usage',
        'remaining deductible pct': 'Remaining deductible',
        'remaining oop max pct': 'Out-of-pocket spending room',
        'total member cost': 'Healthcare spending',
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
    """Generate a human-readable description of why a factor matters."""
    feature_lower = feature_name.lower()
    is_positive = direction == 'increases'
    
    def get_actual_value(feature_key, default_value):
        if member_profile:
            if feature_key in member_profile and member_profile[feature_key] is not None:
                return member_profile[feature_key]
            key_underscore = feature_key.replace(' ', '_')
            if key_underscore in member_profile and member_profile[key_underscore] is not None:
                return member_profile[key_underscore]
        return default_value
    
    if 'engagement' in feature_lower or 'response' in feature_lower:
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
            if is_positive:
                return "Their responsiveness to outreach suggests they'll engage with this offer."
            else:
                return "Their response patterns indicate we may need different messaging approaches."
        else:
            channel = "program"
            channel_desc = "health programs"
        
        if is_positive:
            return f"Their {channel} engagement shows strong receptivity to {channel_desc}."
        else:
            return f"Their {channel} engagement suggests trying different communication channels."
    
    elif feature_lower == 'age' or feature_lower.startswith('age_') or feature_lower.endswith('_age'):
        actual_age = get_actual_value('age', value)
        age_val = int(safe_float(actual_age))
        
        if age_val < 18 or age_val > 120:
            if is_positive:
                return "Their age demographic makes this offer particularly relevant."
            else:
                return "Based on their age group, other offers may be more relevant."
        
        if is_positive:
            return f"At age {age_val}, this member is in a key demographic for this offer's benefits."
        else:
            return f"At age {age_val}, other offers may be more relevant to their life stage."
    
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
    
    elif 'diabetes' in feature_lower:
        actual_val = get_actual_value('has_diabetes', value)
        has_condition = safe_bool(actual_val) or safe_float(actual_val) > 0
        if has_condition and is_positive:
            return "Their diabetes diagnosis makes this program particularly relevant for their care needs."
        elif has_condition:
            return "While they have diabetes, other factors suggest different priorities."
        else:
            return "No diabetes diagnosis on record."
    
    elif 'cardiovascular' in feature_lower or 'heart' in feature_lower:
        actual_val = get_actual_value('has_cardiovascular', value)
        has_condition = safe_bool(actual_val) or safe_float(actual_val) > 0
        if has_condition and is_positive:
            return "Their heart health history makes this program especially beneficial."
        elif has_condition:
            return "Heart health is a consideration, but other factors take priority here."
        else:
            return "No cardiovascular conditions on record."
    
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
    
    else:
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
    for factor in shap_factors[:5]:
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
            html.Div([
                html.Span("Key Decision Factors", style=styles['shap_header'])
            ]),
            html.P("The following member attributes influenced this recommendation:",
                  style={'color': '#6b7280', 'marginBottom': '16px', 'fontSize': '13px', 'margin': '0 0 16px 0'}),
            html.Div(shap_items) if shap_items else html.P(
                "No detailed factor data available for this recommendation.", 
                style={'color': '#9ca3af', 'fontStyle': 'italic', 'fontSize': '13px'}
            )
        ], style=styles['shap_box'])
    ], id=explain_content_id, style={'display': 'none'}) if shap_factors else None
    
    return html.Div([
        # Header with rank, name, and score
        html.Div([
            html.Div([
                html.Span(f"{rec.get('rank', '-')}", style=styles['rank_badge']),
                html.Div([
                    html.H4(rec.get('offer_name', 'Unknown Offer'), style=styles['offer_name']),
                    html.Div(f"ID: {offer_id}", style=styles['offer_id_text'])
                ], style={'marginLeft': '14px'})
            ], style=styles['offer_title_section']),
            html.Div([
                html.Span("RELEVANCE", style=styles['score_label']),
                html.Span(f"{safe_float(rec.get('priority_score')):.0f}", style={'fontSize': '18px', 'fontWeight': '700'})
            ], style=styles['score_badge'])
        ], style=styles['offer_header']),
        
        # AI Recommendation Insight
        html.Div([
            html.Div("AI RECOMMENDATION INSIGHT", style=styles['reasoning_label']),
            html.P(rec.get('llm_reasoning', 'No reasoning available.'), style=styles['reasoning_text'])
        ], style=styles['reasoning_box']) if rec.get('llm_reasoning') else None,
        
        # Explain Button (only if we have SHAP factors)
        html.Button(
            ["View Decision Factors"],
            id=explain_btn_id,
            n_clicks=0,
            style=styles['explain_button']
        ) if shap_factors else None,
        
        # Collapsible Explain Section
        explain_section,
        
        # Feedback section
        html.Div([
            html.Div("REVIEW THIS RECOMMENDATION", style=styles['feedback_label']),
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
                html.Div(id=feedback_status_id)
            ], style=styles['feedback_container']),
        ], style=styles['feedback_section']),
        
        # Comment section
        html.Div([
            html.Div("Add a note (optional)", style={'fontSize': '12px', 'color': '#6b7280', 'marginBottom': '8px'}),
            dcc.Textarea(
                id=feedback_text_id,
                placeholder="Share feedback about this recommendation...",
                style=styles['feedback_textarea']
            ),
            html.Div([
                html.Button(
                    "Submit Note",
                    id=submit_text_id,
                    n_clicks=0,
                    style=styles['submit_feedback_button']
                ),
                html.Div(id=text_status_id, style={'marginLeft': '12px', 'display': 'flex', 'alignItems': 'center'})
            ], style={'display': 'flex', 'alignItems': 'center'})
        ], style=styles['comment_section'])
        
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
    
    is_open = n_clicks % 2 == 1
    
    if is_open:
        return (
            {'display': 'block'},
            ["Hide Decision Factors"],
            styles['explain_button_active']
        )
    else:
        return (
            {'display': 'none'},
            ["View Decision Factors"],
            styles['explain_button']
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
    
    triggered = ctx.triggered[0]
    prop_id = triggered['prop_id']
    
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
            True,
            True
        )
    elif button_type == 'reject-btn' and reject_clicks:
        feedback = 'rejected'
        saved = save_feedback(member_id, offer_id, feedback)
        status_text = "✗ Rejected" + (" (saved)" if saved else " (not saved)")
        return (
            status_text,
            styles['feedback_rejected'],
            True,
            True
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
    
    saved = save_feedback(member_id, offer_id, 'comment', feedback_text.strip())
    status_text = "✓ Comment submitted!" if saved else "⚠️ Comment logged (DB save failed)"
    status_style = styles['feedback_submitted'] if saved else {'color': '#d69e2e', 'fontSize': '14px'}
    
    return (
        status_text,
        status_style,
        True,
        True
    )


if __name__ == '__main__':
    app.run_server(debug=True, host='0.0.0.0', port=8000)

