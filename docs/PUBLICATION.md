# Repository boundary and licences

This is the reviewed source tree for [krystofmincev/finlease](https://github.com/krystofmincev/finlease). The broader local workspace is outside its publication scope. `PUBLIC-FILES.sha256` is the explicit file inventory: reviewed publication contents are limited to its listed paths plus that inventory itself. `.gitignore` helps exclude runtime files; the inventory defines the reviewed boundary.

| Included material | Purpose |
| --- | --- |
| `src/`, `configs/`, `scripts/`, `tests/`, Python/frontend manifests and locks | Byte-identical scientific implementation, schema tools and tests. Three historical JSON artifacts are retained because the unchanged tests read them. |
| `ui/src/`, UI tooling and `ui/dist/` | Typed interface and the exact compiled snapshot for Python-only saved-demo viewing. The compiled UI intentionally belongs to the reviewed source tree. |
| `artifacts/current.json`, current core run and matching TabPFN sidecar | Complete qualified saved product, including the full replay data. |
| Matching pooled run | Selection commitment written before final generation, plus its source-identified development evidence. |
| Selected verification JSON, screenshots and actual browser export | Source-matched measurements, presentation provenance and inspectable results. |
| Separate bootstrap commitment/results/arrays/runner and a path-redacted strict-gate failure | Failed and unadopted methods remain visible. The redundant bootstrap source snapshot is excluded. |
| Public docs and exact third-party licence notices | Self-contained story, methods, setup, boundaries and attribution. |

[Source-files.json](evidence/source-files.json) fingerprints the initial 138 byte-identical copies, including the UI notices. Later reviewed docs/assets are covered by the final inventory. Source/config/lock hashes and the selected result identities are preserved. Public prose is a new summary; it is not a relabelled scientific run or a byte-identical copy of the original internal blueprint.

Excluded: credentials, `.env` files, private correspondence/CVs/salary/career material, employer code, app/chat transcripts, parent research collections, upstream instruction snapshots, pilot/harness/global files, `AGENTS.md`, rollback archives, raw machine-specific logs, pretrained model weights/checkpoints, `.local`, environments, node_modules, caches and temporary QA files. All original local scientific history and private rollback/failure evidence are retained separately; selected public evidence does not claim to contain every historical run.

## Project licence status

No project-level reuse licence has been selected for the original code, authored UI/assets, synthetic fixtures or generated results, and no project `LICENSE` file is added here. The third-party licences below remain separate.

Synthetic data/results and screenshots have authored-fictional provenance; they contain no real customer records. No company approval, affiliation or customer-performance claim is made.

## Third-party material

Built with PriorLabs-TabPFN. The full [Prior Labs License 1.1](licenses/TabPFN-v2.txt) is retained, including enhanced attribution. [Pinned TabPFN details](TABPFN.md) distinguish the Apache-2.0 inference code from the checkpoint licence. Weights are fetched by the optional runtime command and are not included in this repository; no weight-redistribution conclusion is implied.

The retained compiled UI contains third-party runtime code and generated build helpers. Exact notices copied from the installed versions accompany it:

| Package | Licence notice |
| --- | --- |
| React 19.3.1 | [MIT](licenses/ui/react-19.3.1.txt) |
| React DOM 19.3.1 | [MIT](licenses/ui/react-dom-19.3.1.txt) |
| Scheduler 0.28.0 | [MIT](licenses/ui/scheduler-0.28.0.txt) |
| Ajv 8.20.0 | [MIT](licenses/ui/ajv-8.20.0.txt) |
| fast-deep-equal 3.1.3 | [MIT](licenses/ui/fast-deep-equal-3.1.3.txt) |
| fast-uri 3.1.9 | [BSD-3-Clause](licenses/ui/fast-uri-3.1.9.txt) |
| json-schema-traverse 1.0.0 | [MIT](licenses/ui/json-schema-traverse-1.0.0.txt) |
| require-from-string 2.0.2 | [MIT](licenses/ui/require-from-string-2.0.2.txt) |
| Vite 8.3.5 build helpers | [MIT and bundled notices](licenses/ui/vite-8.3.5.txt) |
| Rolldown 1.2.18 build helpers | [MIT](licenses/ui/rolldown-1.2.18.txt), [third-party notices](licenses/ui/rolldown-1.2.18-third-party.txt) |

Python packages and build tools are installed from locks, not bundled as environments or upstream source trees. Their own licences continue to apply. No researched repository's template, instruction text, branding, image or code was imported for the showcase.
