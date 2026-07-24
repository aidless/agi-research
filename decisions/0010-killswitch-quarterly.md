# Decision Record 0010 - Kill Switch tightened (quarterly)

> Date: 2026-07-25
> Status: ADOPTED
> Owner: user + Codex

## Context

TASKBOOK v1 Section 8.4 set Kill Switch on 6-month no-output. User
critique: 6 months is two full quarters; one wrong direction can
consume 4 months before the trigger fires.

## Revised Kill Switch (quarterly)

Every quarter end, run the following check:

1. **Code increment**: did we make a measurable code change in the
   quarter? (commit count > 5, or one runnable experiment added)
2. **Draft section**: did any paper draft progress? (>= 1 new section)
3. **Public footprint**: did we ship anything externally? (PR, post,
   email sent, blog published)

**Failure condition**: any quarter where **all three** are zero
**AND** the user is not making progress on DEC-001 (PhD vs
independent).

**Triggered action**: pause, not kill. Conduct a "pivot or pause"
review. Three options:
- continue on new direction
- pause this project for a quarter
- exit (allows "this project ends")

## Why quarterly, not monthly

Monthly review is too noisy (most months won't have a 95% confidence
directional change). Quarterly gives enough time for real experiments
and writing.

## What changes in TASKBOOK v1.1

- Section 8.4: replace 6-month rule with quarterly rule above.
- Section 9.2: add Kill Switch evaluation to quarterly review
  template (already present).

## Decision deadline

Already adopted - this is a charter-amendment effective immediately.
