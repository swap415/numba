# Investigation records

Create one directory per issue or topic. Copy the outline below into its README
and add numbered scripts as experiments arise. Commands must specify their
working directory and interpreter or build environment. Capture enough setup
for another person to rerun them from a clean checkout; exclude secrets.

```markdown
# <issue or topic>

Problem / expected behavior:
Scope / success criterion:
Baseline revision / patch revision:
Environment / dependencies / build flags / setup commands:
Evidence links and relationships (cause, duplicate, dependency, prior decision):
Known facts / assumptions / failure modes:

## 01 — <question>
Hypothesis / predicted outcomes / next branch:
Script / exact command / working directory:
Observed result (assertion, error, or measurement):
Conclusion / changed understanding / next step:

## Decision and handoff
Cause / alternatives / chosen fix and why:
Blast radius / tradeoffs / remaining uncertainty:
Before and after / tests / limitations:
```

Append a numbered entry for every step, including research and failed attempts.
For benchmarks, retain the script, raw samples, units, warmup, and repetition
count; report baseline and patched results with spread on the same workload.
