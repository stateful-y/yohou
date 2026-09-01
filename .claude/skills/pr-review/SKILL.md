---
name: pr-review
description: "Advanced verified review of the current PR changes (the branch diff against main). Runs correctness, data-contract, design, tests, docstrings, docs, surface, and simplification passes scoped to the diff, then adversarially verifies every finding before reporting. Use whenever the user asks to review the PR, review the branch, review my changes, do a deep review, check this before merge, or wants a code review of work in progress on a yohou branch, even if they do not say 'PR' explicitly."
---

# /pr-review

Verified review of the current PR diff. This is the diff-scoped counterpart of the `/qa` family: `/qa` reviews the repo as it stands and only reports quality findings; this skill reviews what the current branch **changes**, hunts for defects the change introduces, checks what the change breaks elsewhere, and checks that the PR is complete. It reuses the `/qa` chassis wherever the concern is shared, so read `.claude/commands/qa.md` first: its Ground rules (exclusion rule, simplicity rule, severity rubric), finding schema, verification rules, and `reviews/accepted.md` handling all apply here unchanged, with the diff-specific additions below.

Relationship to other tooling:

- `/qa` and its spokes: repo-scoped, report-only. This skill: diff-scoped, report-first with an opt-in fix mode. A quality finding in code the branch does not touch belongs to `/qa`, not here; do not sweep unchanged files for pre-existing issues.
- ruff, ty, rumdl, `--doctest-modules`, CI: deterministic gates. The chassis exclusion rule applies in full. Never report what they enforce; run `just fix` observations through them, not through findings.

## Arguments

`/pr-review [base] [dimensions] [--fix] [--report] [--comment]`

- `base`: git ref to diff against. Default: `$(git merge-base HEAD origin/main)`. A PR number resolves via `gh pr view <n> --json baseRefName`.
- `dimensions`: comma-separated subset of `correctness,contract,design,tests,docstrings,docs,surface,simplify,completeness`, or omitted for all.
- `--fix`: after reporting, apply the confirmed `high` and `medium` findings to the working tree, one commit-sized batch per dimension, and rerun `just fix` and the touched tests. Without the flag, report only. Never fix `low` or unconfirmed findings under `--fix`; list them for the user.
- `--report`: also write the report to `reviews/<YYYY-MM-DD>-pr-<branch-slug>.md` in the chassis report format. Default is chat-only, because PR reviews are ephemeral and `reviews/` holds the committed `/qa` baseline.
- `--comment`: post confirmed findings as inline comments on the GitHub PR via `gh` (only when a PR exists for the branch; confirm with the user first, posting is outward-facing).

## Severity and verdict

Use the chassis severity rubric verbatim (`high` = data-contract violation, correctness bug, or false claim; `medium` = cross-family or code-vs-convention inconsistency, misleading structure per the chassis closed list; `low` = judgment-call improvements and all other simplifications). Two diff-specific clarifications:

- A defect the diff **introduces** is judged at full severity. A pre-existing defect the diff merely sits next to is out of scope (route it to `/qa`); a pre-existing defect the diff makes worse or copies into new code is in scope.
- The review ends with a verdict: `REQUEST_CHANGES` if any confirmed `high` finding survives, `COMMENT` if only `medium`/`low` survive, `APPROVE` if nothing survives verification.

## Phase 0: diff manifests

Run these before any review. If a script fails, stop and report; never substitute guesses.

**Diff manifest (`DIFF`).**

```bash
base=${1:-$(git merge-base HEAD origin/main)}
git log --oneline "$base"..HEAD
git diff "$base"..HEAD --stat -- . ':(exclude)uv.lock' ':(exclude)*.svg' ':(exclude)*.png'
git diff "$base"..HEAD --name-only -- . ':(exclude)uv.lock'
```

The changed non-lock line count picks the orchestration tier (below). Keep the changed-file list as `CHANGED_FILES`; it plays the role of the chassis incremental manifest, except here it defines the whole scope, not a filter on a full sweep.

**Impact manifest (`IMPACT`).** For every public or package-internal symbol the diff adds, removes, or changes the signature of (read the diff hunks to list them), find the callers the diff does not touch:

```bash
git diff "$base"..HEAD -U0 -- 'src/**/*.py' | grep -E '^[+-] *(def |class |[A-Z_]+ =)' | sort -u
# then for each changed symbol NAME:
grep -rn --include='*.py' '\bNAME\b' src/ tests/ docs/
```

Hits in files outside `CHANGED_FILES` are the impact set: the places a signature or behavior change can break without the diff showing it. Hand this set to the correctness reviewer.

**Completeness manifest (`COMPLETENESS`).** Answer each with yes/no plus the evidence, mechanically:

- Every changed `src/yohou/` module has a changed or existing test exercising the changed behavior (name the test file; `tests/test_common.py` counts only when the change is inside an estimator the systematic checks reach).
- Every new estimator class is discoverable by `all_estimators()` and wired into the matching `_yield_yohou_*_checks` suite, per the family's `create-yohou-*` skill.
- Every new or signature-changed public symbol is exported where the family exports it, has a numpydoc docstring, and appears in the docs API surface.
- Behavior changes that alter a documented claim have the doc page in the diff.
- Commits follow conventional-commit types (git-cliff builds the changelog from them; a `feat` labeled `chore` disappears from the release notes).

Every "no" is automatically a finding (dimension `completeness`, severity `medium`, category matching the bullet). The check is diff-relative: it never demands work the diff does not touch.

## Review passes

One charter per dimension. For the quality dimensions, do not restate the `/qa` spoke charters: the reviewer reads the spoke file and applies its categories **restricted to `CHANGED_FILES` and the diff hunks**. The diff-native dimensions are defined here.

| Dimension | Charter source | Scope |
| --- | --- | --- |
| correctness | below | diff hunks + `IMPACT` set |
| contract | below | diff hunks |
| design | `.claude/commands/qa-design.md` categories | changed modules (skip the global consistency unit; run `naming-consistency` only over symbols the diff adds) |
| tests | `.claude/commands/qa-tests.md` categories | changed test files + tests paired with changed src modules |
| docstrings | `.claude/commands/qa-docstrings.md` categories | changed `.py` files |
| docs | `.claude/commands/qa-docs.md` categories | changed `docs/` files, plus pages citing changed symbols |
| surface | `.claude/commands/qa-surface.md` categories | exports the diff adds or removes |
| simplify | qa-source `needless-complexity` + `comment-altitude` | diff hunks only (new code held to the simplicity rule; do not flag surrounding old code) |
| completeness | Phase 0 manifest | the PR as a unit |

### correctness charter (diff-native)

```text
You are a correctness reviewer for a Yohou PR (chassis rules apply; return findings in the hub JSON schema, dimension "correctness").
DIFF: {hunks} | IMPACT: {untouched callers of changed symbols}

Read every hunk with its surrounding function, then the IMPACT set. Findings per category:

### introduced-bug
- Logic errors in new or modified code: inverted conditions, off-by-one on horizons/windows, wrong column referenced, wrong join keys, mutation of caller-owned frames.
- Edge inputs the new code mishandles: empty frames, single-row groups, a panel with one group, horizon 1, all-null columns, non-contiguous time.
- Error paths: exceptions swallowed, misleading messages, validation after partial mutation (qa-source error-handling, judged on new code).

### behavior-regression
- A changed function now returns something different for inputs an IMPACT caller sends; cite the caller line that breaks.
- A default value, tag, or fitted-attribute name change no caller was updated for.

### concurrency-and-state
- Fitted state mutated in predict/transform; shared mutable defaults; caches keyed too narrowly.

### test-honesty
- A test changed in this diff that now passes for the wrong reason: assertion weakened to make the diff green, tolerance widened without justification, xfail added where a fix belongs.

Rules: cite file:line into the NEW side of the diff; judge only what the diff introduces or breaks; at most 10 findings, highest value first.
```

### contract charter (diff-native)

```text
You are a data-contract reviewer for a Yohou PR (chassis rules apply; dimension "contract", severity high unless stated).
DIFF: {hunks}

Check every hunk against the contract: polars-native (no pandas round-trips), mandatory "time" column, group__column panel naming, vintage_time + time in prediction outputs, learned state suffixed with _ and set only in fit. Also:
- New parameters must appear in _parameter_constraints with the right constraint.
- Panel dispatch: new code paths must handle the panel case or raise explicitly, never silently compute over pooled rows.
- Output schema: new columns documented, column order and dtypes consistent with the family's other estimators (medium if merely inconsistent).

Rules: cite file:line; at most 8 findings.
```

## Orchestration

Invoking this skill is the user's explicit opt-in to multi-agent orchestration. Pick the tier from the non-lock changed line count in `DIFF`:

- **Small (< ~200 lines).** No fan-out. Review every dimension yourself in one pass over the diff, then spawn ONE verifier subagent with the whole finding batch. Cheap PRs do not justify a workflow.
- **Medium (~200-1000 lines).** Workflow: one reviewer agent per selected dimension, then batched verifiers.
- **Large (> ~1000 lines).** Workflow: split the per-file dimensions into units so no reviewer reads more than ~1500-2000 lines (chassis rule 3); `correctness` splits by file group, each unit still receiving the full `IMPACT` slice for its symbols.

All three chassis orchestration rules are load-bearing here too: batch the verifiers (~8 findings per verifier, one verdict per id), checkpoint raw findings to `reviews/.pr-review-checkpoint.json` before verification, and size units by lines. Model tiering follows the hub: reviewers on `sonnet`, verifiers inherit the main-loop model. The hub's Workflow template applies with `unitJobs` built from the table above and `consistencyPrompts` empty; reuse it rather than writing a new script shape.

Pass the accepted entries from `reviews/accepted.md` for `CHANGED_FILES` into the `simplify` and `design` reviewer prompts (chassis simplicity rule); a reviewer skips only findings equivalent to an accepted entry.

## Verification

Chassis verification rules apply verbatim: refute-first, excerpt-first with escalation, one verdict per id, verifier severity wins, `unconfirmed` downgrades one level, and simplification findings pass only the three-part test (named replacement, behavior preserved, countable reduction) with `unconfirmed` treated as refuted. Two diff-specific additions:

- `behavior-regression` and `introduced-bug` verifiers must state the concrete failing input in the citation ("empty group after filter at line N yields ...") or return `unconfirmed`; a plausible-sounding bug without a failing input is how false positives survive.
- `completeness` findings are manifest-derived and skip verification; they are confirmed by construction.

## Report

Present in chat (and to `--report`/`--comment` targets when asked):

```markdown
# PR review: <branch> (<n> commits, +A/-D over <base-sha>)
Verdict: <APPROVE | COMMENT | REQUEST_CHANGES>
Findings: N confirmed (H high, M medium, L low), U unconfirmed, R refuted in verification
Completeness: <the manifest bullets, pass/fail each>

## High / ## Medium / ## Low
| ID | Category | File | Claim | Evidence | Suggestion |

## Unconfirmed
<kept per chassis rules, downgraded, marked>

## Refuted (appendix, ids only)
```

Lead the chat message with the verdict and the high findings in prose, not the table. If nothing survives, say what was reviewed and what was not (dimensions skipped, impact set size), per the clean-review discipline: an empty report with unstated scope is indistinguishable from a shallow one.

Then stop. Without `--fix`, do not modify files; the deliverable is the review. With `--fix`, apply as specified under Arguments, rerun the touched tests, and report what changed and what was deliberately left. Do not edit `reviews/accepted.md`, do not open OpenSpec changes, do not push or comment on GitHub without the flag.

If a run halts before synthesis, recover from `reviews/.pr-review-checkpoint.json`, mark everything `UNVERIFIED`, state which phases did not run, and delete the checkpoint once the report is written.
