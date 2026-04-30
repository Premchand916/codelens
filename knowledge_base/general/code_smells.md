# Common Code Smells

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
