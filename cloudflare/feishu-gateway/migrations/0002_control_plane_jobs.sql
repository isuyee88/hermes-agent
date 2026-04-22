CREATE TABLE IF NOT EXISTS control_plane_jobs (
  job_name TEXT PRIMARY KEY,
  next_run_at INTEGER NOT NULL DEFAULT 0,
  lease_until INTEGER NOT NULL DEFAULT 0,
  last_run_at INTEGER,
  last_run_status TEXT,
  last_run_token TEXT,
  last_error TEXT,
  updated_at INTEGER NOT NULL DEFAULT (unixepoch())
);

CREATE INDEX IF NOT EXISTS idx_control_plane_jobs_due
  ON control_plane_jobs(next_run_at, lease_until);
