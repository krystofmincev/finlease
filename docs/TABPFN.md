# Pinned pretrained TabPFN V2

Built with PriorLabs-TabPFN. This comparison uses pretrained point predictions, without task fine-tuning, a posterior adapter or cautious/Thompson draws. [Results](RESULTS.md) disclose fitting-label, preprocessing/support and external pretraining differences.

| Setting | Pinned value |
| --- | --- |
| Package / Torch | 9.2.0 / 2.14.1 in `uv.lock`'s optional challenger dependencies |
| Checkpoint repository | `Prior-Labs/TabPFN-v2-clf` |
| Revision | `275f19d2add9c2c0de38f9ebaded3856599fa2e7` |
| Filename / size | `tabpfn-v2-classifier-v2_default.ckpt` / 29,009,539 bytes |
| SHA256 | `cf8c519c01eaf1613ee91239006d57b1c806ff5f23ac1aeb1315ba1015210e49` |
| Context / device | 512 stable-ID fitting applicants per world; CPU, float32, one estimator |
| Shared history | Each world's frozen full-fit transform and advisory support history |
| Seed / query chunk | 20261109 / 96 |
| Task search / calibration labels | 0 / 0 |

Weights are not included in this package. After `make reproduce`, the documented command is:

```sh
uv run --extra challenger lease-lab benchmark-tabpfn
```

It installs optional dependencies, then reuses a SHA-verified checkpoint or downloads it from the [pinned official URL](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/resolve/275f19d2add9c2c0de38f9ebaded3856599fa2e7/tabpfn-v2-classifier-v2_default.ckpt) into `.local/tabpfn-models/`. A mismatched download fails rather than silently selecting different weights. The URL returned HTTP 200 after redirect during packaging inspection; that was a header check, not a new benchmark or model download. Access and network availability may change.

Read the retained [Prior Labs License 1.1](licenses/TabPFN-v2.txt), also available at the [exact upstream revision](https://huggingface.co/Prior-Labs/TabPFN-v2-clf/resolve/275f19d2add9c2c0de38f9ebaded3856599fa2e7/LICENSE.txt), before obtaining or redistributing weights. It includes enhanced attribution. The [9.2.0 inference-code licence](https://github.com/PriorLabs/TabPFN/blob/v9.2.0/LICENSE) is Apache-2.0; code and checkpoint licences are distinct. The package makes no new grant of rights to weights or dependencies. Weight redistribution is outside this candidate.

After checkpoint preparation, network connections are blocked during model import, fitting and prediction. Inference has a native 600-second alarm and a 1.5 GB RSS target checked after prediction calls; the RSS check is not an OS hard memory limit. Serving reads only strict JSON and imports neither Torch nor TabPFN.

The sidecar must match the core's source, final factual cohort, selected experiment/freeze, feature, context-ID/data and matrix identities. Missing/stale/incomplete rows fail qualification. Saved evaluator bootstrap intervals for policy value represent sampling variation, not transformer epistemic uncertainty.
