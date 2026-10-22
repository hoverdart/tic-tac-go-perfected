-- Run against the site's database to refresh the recorded-outcome metrics.
-- This does not measure failed push attempts followed by a successful fallback.
BEGIN READ ONLY;
SET LOCAL statement_timeout = '30s';

SELECT
  count(*) AS stored_daily_boards,
  count(*) FILTER (WHERE status = 'solved') AS solved_daily_boards,
  count(*) FILTER (WHERE status = 'solved' AND elapsed_ms IS NULL) AS solved_without_timing
FROM daily_solutions;

SELECT
  solver_name,
  count(*) AS stored_rows,
  count(*) FILTER (WHERE status = 'solved') AS solved_rows,
  count(elapsed_ms) FILTER (WHERE status = 'solved') AS timed_solved_rows,
  count(*) FILTER (WHERE status = 'solved' AND elapsed_ms <= 30000) AS solved_within_30_seconds,
  percentile_cont(0.5) WITHIN GROUP (ORDER BY elapsed_ms / 1000.0)
    FILTER (WHERE status = 'solved') AS median_seconds,
  avg(elapsed_ms / 1000.0) FILTER (WHERE status = 'solved') AS mean_seconds,
  max(elapsed_ms / 1000.0) FILTER (WHERE status = 'solved') AS max_seconds,
  sum(states_checked) FILTER (WHERE status = 'solved') AS total_reported_states,
  percentile_cont(0.5) WITHIN GROUP (ORDER BY states_checked)
    FILTER (WHERE status = 'solved') AS median_reported_states
FROM daily_solutions
GROUP BY solver_name
ORDER BY solver_name;

ROLLBACK;
