-- Bronze: raw members extracts landed as JSON in the UC Volume, ingested
-- incrementally with Auto Loader (read_files on STREAM). Schema is inferred and
-- evolved; malformed fields are captured in _rescued_data rather than dropped.
CREATE OR REFRESH STREAMING TABLE bronze_members
COMMENT 'Raw members extract (synthetic) ingested with Auto Loader from the raw_landing volume. One row per source record per landed batch.'
TBLPROPERTIES ('quality' = 'bronze')
AS SELECT
  *,
  _metadata.file_path AS _source_file,
  current_timestamp() AS _ingested_at
FROM STREAM read_files(
  '${landing_path}/members/',
  format => 'json',
  rescuedDataColumn => '_rescued_data'
);
