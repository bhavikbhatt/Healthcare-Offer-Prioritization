-- Bronze: raw engagement_history extracts landed as JSON in the UC Volume, ingested
-- incrementally with Auto Loader (read_files on STREAM). Schema is inferred and
-- evolved; malformed fields are captured in _rescued_data rather than dropped.
CREATE OR REFRESH STREAMING TABLE bronze_engagement_history
COMMENT 'Raw engagement_history extract (synthetic) ingested with Auto Loader from the raw_landing volume. One row per source record per landed batch.'
TBLPROPERTIES ('quality' = 'bronze')
AS SELECT
  *,
  _metadata.file_path AS _source_file,
  current_timestamp() AS _ingested_at
FROM STREAM read_files(
  '${landing_path}/engagement_history/',
  format => 'json',
  rescuedDataColumn => '_rescued_data'
);
