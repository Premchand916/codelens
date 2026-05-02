# Credential Handling

## Environment Variables Are Not Enough Alone
`.env` files must be in `.gitignore`. Verify before every commit.
Use `python-dotenv` for local dev, proper secrets manager in prod.

## Rotate Compromised Secrets Immediately
If a key is committed to git — even for 1 second — treat it as compromised.
Git history preserves it. Rotate the key, don't just delete the commit.

## Minimum Privilege
API keys should have only the permissions they need.
A webhook validation key should not also have write access.

## Never Log Secrets
Check all logger.info/debug calls that include request headers or bodies.
Redact Authorization headers before logging.

## Token Expiry
Prefer short-lived tokens with refresh over long-lived static keys.