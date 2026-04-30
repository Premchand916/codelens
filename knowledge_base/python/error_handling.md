# Python Error Handling Best Practices

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
