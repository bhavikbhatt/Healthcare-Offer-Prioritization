-- Silver: typed member outreach / digital engagement events.
CREATE OR REFRESH STREAMING TABLE silver_engagement_history (
  CONSTRAINT valid_engagement_id EXPECT (engagement_id IS NOT NULL) ON VIOLATION DROP ROW,
  CONSTRAINT valid_member_id EXPECT (member_id IS NOT NULL) ON VIOLATION DROP ROW,
  CONSTRAINT valid_channel EXPECT (channel IN ('email', 'app', 'phone', 'mail', 'portal', 'sms')) ON VIOLATION DROP ROW
)
COMMENT 'Member engagement events (synthetic) across email, app, phone, mail, portal and SMS, with response flag. Grain: one row per engagement event.'
TBLPROPERTIES ('quality' = 'silver')
AS SELECT
  engagement_id,
  member_id,
  CAST(engagement_date AS TIMESTAMP) AS engagement_date,
  channel,
  engagement_type,
  CAST(response_flag AS BOOLEAN) AS response_flag,
  CAST(session_duration_sec AS INT) AS session_duration_sec,
  offer_id,
  batch_id,
  _ingested_at
FROM STREAM(bronze_engagement_history);
