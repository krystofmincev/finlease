# Open, check or reproduce

Clone the [repository](https://github.com/krystofmincev/finlease) with Git:

```sh
git clone https://github.com/krystofmincev/finlease.git
cd finlease
```

Alternatively, choose **Code → Download ZIP** on GitHub, extract it and open the project folder.

Run commands from the folder containing `pyproject.toml`, `Makefile`, `src/`, `ui/` and `artifacts/`. Keep that layout: configuration and saved results are project-relative.

## View the saved product

Requirements: Python 3.12, [uv](https://docs.astral.sh/uv/getting-started/installation/) and Make. The implementation was verified on macOS arm64; other platforms are not claimed as tested.

```sh
uv sync --frozen
make serve
```

Wait for the terminal's `Uvicorn running` message, then open [http://127.0.0.1:43187](http://127.0.0.1:43187). The saved bundle and compiled UI are included. This path installs base Python dependencies, reads saved JSON and computes decisions; it does not install Torch, download weights or rerun experiments. Stop with Ctrl-C. If the port is occupied, stop your earlier instance before starting another.

No API key, account, parent-workspace file or global agent helper is required. Initial dependency installation needs network access. The screenshot and [saved results](RESULTS.md) can be inspected without installing anything.

## Build the UI and run checks

Add Node 24 and npm. Both dependency locks are retained.

```sh
make setup
npm run build --prefix ui
make check
```

`make setup` uses `uv sync --frozen` and `npm ci --ignore-scripts`; its npm cache is local to the project. The build checks TypeScript and compiles the UI. `make check` runs Python tests, Ruff, real-response Ajv contracts, TypeScript and strict saved-comparison/serving-boundary checks. Small numerical and fit fixtures run during tests; the full benchmark is not regenerated. The required saved TabPFN sidecar is already included, so these checks do not need Torch or weights.

## Repeat the complete comparison

Use a separate copy of this package to preserve the submitted results. Full reproduction writes deterministic run paths and the active pointer. A POSIX environment is required by the current benchmark's `resource` and `SIGALRM` controls; native Windows full reproduction is not supported by these commands.

After `make setup`, run the complete sequence:

```sh
make reproduce
uv run --extra challenger lease-lab benchmark-tabpfn
make check
```

`make reproduce` generates both synthetic worlds, performs bounded development, commits both selections before final generation, runs coverage/reference/replay work, regenerates API contracts and builds the UI. The second command installs the locked optional TabPFN/Torch dependencies and, when absent, downloads the pinned 29 MB checkpoint. It then runs pretrained inference locally. See [the checkpoint, licence and download details](TABPFN.md).

Completion requires the real pretrained sidecar and all five point-method prediction/greedy rows per world. The core alone is incomplete; `--core-only` is a historical repair-checkpoint option, not a shortcut for the finished comparison. Repeating the committed final cohort does not create a new untouched test set. Do not use its published outcomes to choose new settings and present them as fresh confirmation.

The Makefile and offline CLI set one-thread numerical controls. Timings and platform-dependent floating-point output can vary; content/run identities cover the declared source/config/locks, not a promise that every timestamp or output byte repeats. Original measurements are in [verification](VERIFICATION.md). Packaging verification does not claim a fresh full scientific rerun unless explicitly recorded there.

Built with PriorLabs-TabPFN.
