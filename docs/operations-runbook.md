# Operations Runbook

## Telemetry contract

Track these fields for each request:

- `request_id`
- `provider`
- `model`
- `latency_ms`
- `retry_attempts`
- `fallback_attempts` (router-level)
- `token_usage` and `image_usage` when available

## Suggested alerts

- Error rate > 2% over 5 minutes.
- Fallback rate > 10% over 5 minutes.
- P95 latency > 3000 ms for core endpoints.
- Sudden drop in token/image usage reporting (instrumentation failure).

## Incident playbooks

### Provider outage

1. Confirm elevated provider-specific errors.
2. Enable/force router fallback to healthy providers.
3. Reduce max latency/cost caps if needed.
4. Communicate degraded behavior and ETA.

### Authentication failure

1. Verify env keys and secret manager sync.
2. Rotate affected API key.
3. Re-run preflight and integration smoke checks.

### Model deprecation / access loss

1. Confirm model availability in provider console/API.
2. Switch to approved fallback model.
3. Update docs/tests model list and release notes.
