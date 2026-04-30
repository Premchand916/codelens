# Python Security Best Practices

## Never Hardcode Credentials
API keys, passwords, tokens must come from environment variables, never source code.
BAD: API_KEY = "sk-abc123"
GOOD: API_KEY = os.environ["API_KEY"]

## Validate All External Input
Never trust data from webhooks, APIs, or users. Validate schema with Pydantic.
Reject unexpected fields. Set maximum length limits on string inputs.

## Parameterize Database Queries
Never concatenate user input into SQL strings — use parameterized queries.

## Secrets in Logs
Scrub credentials before logging. Check f-strings that include request bodies.

## Dependency Security
Pin dependency versions in requirements.txt.
Run pip audit regularly to catch known CVEs.
