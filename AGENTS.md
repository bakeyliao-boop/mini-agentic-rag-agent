# Repository Working Rules

## PLAN maintenance

`PLAN.md` is the source of truth for local development progress.

After every implementation, bug fix, refactor, dependency or environment change:

1. Compare the current code and tests with every affected `PLAN.md` checklist item.
2. Check an item only when the current local checkout implements it and its acceptance evidence is valid.
3. Append a dated row to the progress record describing what changed and what was verified.
4. Update the final “下一步” section to the next single actionable item.
5. Run `uv run pytest -q`; record failures or blockers instead of claiming completion.

After every `pull`, `merge`, `rebase`, or `cherry-pick`:

1. Inspect the newly introduced commits and file changes.
2. Reconcile `PLAN.md` checkboxes, acceptance state, progress record, and next step.
3. Do not count work that exists only on an unmerged remote branch as locally complete.

`PLAN.md` must remain eligible for Git tracking, be committed with project
changes, and must not be added back to `.gitignore`.

Local-only assets under `knowledge/`, `data/`, and `.env` remain ignored. Checklist
items that depend on those assets may be marked complete only after their presence
and acceptance criteria have been verified in the current workspace.
