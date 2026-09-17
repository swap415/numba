# Working principles

Treat code and prose as things others must trust, review, and maintain.
Prefer the smallest correct design. Reuse idioms, helpers, and infrastructure;
prefer deletion and explicit data flow over new abstractions and hidden state.
Earn every line. Explain only what clear code cannot express.
Follow [tooling conventions](TOOLING.md); update them when evidence or user
feedback establishes a better command. Keep task-specific details in the record.

## Process

1. **Frame.** State the problem, expected behavior, scope, and success criterion.
   Separate observations from assumptions. List likely failure modes first.
2. **Study.** Read project guidance, docs, nearby code, tests, and history
   (`git log`, `git blame`). Trace relevant issues, PRs, reviews, and upstream
   discussions; record links and how they relate. Look for existing solutions
   and current techniques. Infer taste from accepted changes and review reasons,
   not isolated examples. Follow leads that could change the approach; stop
   when the evidence supports a decision and remaining uncertainty is explicit.
3. **Reproduce.** Save the smallest failing example outside production code.
   Prefer assertions. Record the revision, environment, exact command, and
   expected versus actual result. Establish a baseline before editing.
4. **Isolate.** Before each experiment, state the hypothesis, command, outcomes
   that support or falsify it, and next branch. Change one variable at a time.
   Record the result and revised understanding. Backtrack when evidence fails.
   Explain the likely cause before patching; work at the user's pace.
5. **Decide.** Compare no change, reuse, and the smallest fix. Trace consequences
   through callers, shared invariants, public contracts, downstream users, and
   future maintenance. Name affected platforms and versions, tradeoffs, and
   reversal cost. Check whether fixing an earlier layer removes the problem.
   Scale investigation and validation to the blast radius.
6. **Patch.** Address one cause. Match local idioms and keep each change locally
   understandable. Separate behavior changes from formatting and cleanup.
   Preserve unrelated work. Add regression coverage for every behavior change.
7. **Verify.** Show the regression fails before and passes after; run affected
   tests and required project checks. For performance claims, warm up, repeat,
   compare the same workload against baseline, report spread and sample count,
   and verify the work was not optimized away. Optimize measured bottlenecks.
   Review the diff as a maintainer: justify each line, report limitations, and
   hand off the cause, decision, evidence, and remaining risks concisely.

## Reproducible record

Use `investigations/<issue-or-topic>/` with a README and numbered scripts such
as `01_repro.py`, `02_isolate.py`, and `03_bench.py`. Aim for 5–20 readable lines
per experiment; do not compress code or build a framework to meet a quota.
Keep every investigation step, command, result, and consequential dead end.
Preserve earlier experiments; add the next step instead of overwriting evidence.
Follow [the record template](investigations/README.md). Keep investigation
artifacts separate from the upstream patch unless they belong in the project.

## Numba

Follow the parent AGENTS.md, [contribution guide](../docs/source/developer/contributing.rst),
and [coding guidelines](../docs/source/developer/coding_guidelines.rst).
Use `@jit` in new examples. Keep commit subjects short and lowercase.
Read and follow the [AI policy](https://numba.readthedocs.io/en/stable/reference/ai_tools_policy.html):
human review before sharing; disclose assistance (e.g. `Assisted-by: Codex`);
do not use AI to fix `good first issue` issues. Retain the PR template's AI
declaration. The human contributor must understand and own the contribution.
