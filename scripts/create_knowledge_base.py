from pathlib import Path

docs = {
    "python/error_handling.md": """# Python Error Handling Best Practices

## Catch Specific Exceptions
Never use bare `except:` — it catches SystemExit and KeyboardInterrupt.
Catch the most specific exception type possible.

BAD: except: pass
GOOD: except ValueError as e: logger.error(f"Invalid value: {e}"); raise

## Always Log Before Re-raising
Log the exception with context before re-raising so the trace is preserved.
Use `raise` not `raise e` — the latter resets the traceback.

## Never Swallow Exceptions Silently
except Exception: pass hides bugs. At minimum, log them.
If you intentionally ignore an exception, add a comment explaining why.

## Use Finally for Cleanup
finally runs whether or not an exception occurred — use for resource cleanup.
Prefer context managers (with statement) over manual try/finally.

## Custom Exceptions
Define custom exception classes for domain errors.
Inherit from Exception, not BaseException.
""",

    "python/async_patterns.md": """# Python Async Best Practices

## Never Block the Event Loop
Blocking calls (time.sleep, requests.get) inside async functions freeze all coroutines.
Use asyncio.sleep and httpx (async HTTP) instead.

BAD: async def fetch(): time.sleep(1); requests.get(url)
GOOD: async def fetch(): await asyncio.sleep(1); await client.get(url)

## Use asyncio.gather for Concurrent Tasks
Run independent coroutines concurrently, not sequentially.
Sequential: result_a = await task_a(); result_b = await task_b()
Concurrent: result_a, result_b = await asyncio.gather(task_a(), task_b())

## Timeout Everything
Network calls can hang. Always wrap with asyncio.wait_for.

## Async Context Managers
Use async with for resources that need async setup/teardown.
""",

    "python/security_practices.md": """# Python Security Best Practices

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
""",

    "general/code_smells.md": """# Common Code Smells

## Long Functions
Functions over 30 lines usually do too many things. Extract into smaller functions.
Each function should have one clear responsibility.

## Magic Numbers
Unnamed numeric constants make code unreadable.
BAD: if len(chunks) > 16: split()
GOOD: MAX_CHUNKS = 16; if len(chunks) > MAX_CHUNKS: split()

## Deep Nesting
More than 3 levels of indentation signals missing abstraction.
Use early returns (guard clauses) to flatten nesting.

## Duplicate Code
Copy-pasted logic means two places to fix bugs. Extract to a shared function.

## Boolean Traps
process(True, False, True) — what do these mean?
Use keyword arguments or enums instead of positional booleans.
""",

    "general/naming_conventions.md": """# Naming Conventions

## Be Descriptive, Not Terse
user_authentication_token beats uat or token.
Exception: loop variables (i, j) and well-known math variables (x, y).

## Functions Should Be Verbs
get_user(), validate_input(), send_review_comment().
Not user(), input(), comment().

## Booleans Should Be Questions
is_valid, has_permission, should_retry.
Not valid, permission, retry.

## Constants Are UPPER_SNAKE_CASE
MAX_RETRY_COUNT, DEFAULT_TIMEOUT_SECONDS.

## Avoid Abbreviations Unless Universal
config, db, url, id are fine. usr, mgr, proc are not.
""",

    "security/credential_handling.md": """# Credential Handling

## Environment Variables Are Not Enough Alone
.env files must be in .gitignore. Verify before every commit.
Use python-dotenv for local dev, proper secrets manager in prod.

## Rotate Compromised Secrets Immediately
If a key is committed to git even for 1 second — treat it as compromised.
Git history preserves it. Rotate the key, do not just delete the commit.

## Minimum Privilege
API keys should have only the permissions they need.
A webhook validation key should not also have write access.

## Never Log Secrets
Check all logger.info/debug calls that include request headers or bodies.
Redact Authorization headers before logging.

## Token Expiry
Prefer short-lived tokens with refresh over long-lived static keys.
""",
}

kb_root = Path("knowledge_base")

for relative_path, content in docs.items():
    file_path = kb_root / relative_path
    file_path.parent.mkdir(parents=True, exist_ok=True)
    file_path.write_text(content, encoding="utf-8")
    print(f"Written: {file_path} ({len(content)} chars)")

print(f"\nDone. {len(docs)} files written.")