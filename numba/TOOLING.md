# Tooling conventions

Keep proven, reusable lessons here: preferred command, reason, and scope.
Record failed attempts and exact versions in the investigation, not this guide.
Check existing instructions and workflows before inventing environment setup.

## Python and conda

- Prefer `conda run -n <env> <command>` over absolute interpreter paths or shell
  activation. This is the user's convention and applies the environment setup.
- For long jobs, use `conda run --no-capture-output -n <env> <command>` to stream
  progress. Use `python -u` if Python itself buffers output.
- Set the working directory explicitly. Run Numba commands from the repository
  root (`git rev-parse --show-toplevel`), not its `numba/` package directory:
  the latter can shadow standard-library modules such as `types`.
- For source-checkout experiments, prefer `python -m <module>` from that root.
  Subprocess tests may also need the checkout on `PYTHONPATH` if it is not
  installed in the environment; a parent's working directory is not enough.
- Check versions and import paths before tests. Use a compatible existing
  environment or create one dedicated to the task; preserve unrelated envs.
- Install through `conda run -n <env> python -m pip`, so pip and Python agree.
  Use the dependency source and constraints in the current project workflow.
  Numba's development wheel index is currently
  `https://pypi.anaconda.org/numba/label/dev/simple`; use `--pre -i <index>` for
  llvmlite development builds. Check the workflow before choosing a version.
- Treat import/version errors as setup failures. Fix the environment before
  drawing conclusions about the patch; do not bypass dependency checks.

## Commands and recovery

- Prefer `rg` for discovery; narrow paths and output to the question at hand.
- Keep independent commands separate and parallelize reads where useful.
  Keep installs, edits, and other dependent steps sequential.
- After an interrupted command, inspect its process and resulting state before
  retrying: interruption does not prove that nothing happened.
- Distinguish sandbox/network failures from missing packages or code failures.
  Use the approval mechanism when required; changing tools is not a workaround.
- Save reusable experiments as small scripts with exact rerun commands. Use
  `apply_patch` for file edits and inspect the resulting diff.
- Match CI's check commands and file selection. Passing a file explicitly can
  bypass normal discovery filters; Numba's Flake8 run does not discover `.pyi`
  stubs. Validate those with the project's type and stub checks.
- Check recursive lint scope in a developer checkout: local virtualenvs can
  pull third-party packages into the scan. Exclude local environment directories
  explicitly; do not change project lint policy to silence dependency code.
