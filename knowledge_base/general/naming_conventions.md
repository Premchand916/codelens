# Naming Conventions

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
