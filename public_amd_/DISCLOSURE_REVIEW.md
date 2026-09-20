# Disclosure Review

Review scope: `public_amd_demo/` only. This record is a static pre-publication review; run it again after any later edit or after executing the notebook.

| Check | Status | Evidence / required action |
|---|---|---|
| Original project name, paper title, author, organization, or email | PASS | None intentionally included. |
| Private directories or local absolute paths | PASS | Source uses no filesystem reads or hard-coded local paths. |
| Private dataset names, samples, fields, checkpoints, or results | PASS | Synthetic tensors are generated at runtime; no artifact files are included. |
| Original model, function, class, or configuration names | PASS | Demo defines only generic standalone helpers and `TinyClassifier`. |
| API keys, tokens, passwords, IP addresses, or server names | PASS | No network client or credential configuration is present. |
| Import from parent directory or original repository | PASS | Imports are limited to Python standard library, PyTorch, and local `src.amd_runtime`. |
| Copied research comments, custom loss, algorithm, or experimental logic | PASS | The model is a standard two-layer MLP with cross-entropy on synthetic labels. |
| Fabricated runtime output or performance data | PASS | Notebook cells contain no saved outputs; metrics are computed only when run. |
| ROCm wording accuracy | PASS | Documentation explains that ROCm PyTorch may use the `torch.cuda` compatibility namespace. |
| Post-execution output review | WARNING | Before publication, inspect executed cell outputs for machine-identifying names or paths; clear outputs if uncertain. |

Result: no FAIL items in the generated static content. Do not publish if a later edit or executed output introduces a FAIL item.

