# GeoSemID Model-Role Contract

The completed manuscript must distinguish the following model roles.

| Role | Execution time | Function | Status |
|---|---|---|---|
| GeoSemID semantic-code generator | Offline | Fine-tuned local Qwen model in the 8B class that maps POI metadata to a constrained semantic ID | exact checkpoint and SFT details require author confirmation |
| Qwen-Plus reverse-reasoning teacher | Training-data preparation only | Generates reverse-reasoning supervision from target-conditioned inputs | author-confirmed role; exact prompt and filtering protocol require confirmation |
| Predictor/Ranker | Training and test inference | Fine-tuned local small model trained on the teacher-generated reverse-reasoning data and used for deployed candidate-constrained inference | exact backbone and LoRA configuration require confirmation |

The causal training path is:

Qwen-Plus teacher -> reverse-reasoning training data -> fine-tuned local
Predictor/Ranker -> test-time inference.

This contract does not establish that Qwen-Plus generated GeoSemID semantic-code
labels. GeoSemID supervision remains a separate unresolved source until the
author provides the label construction and validation procedure.

The current local CoMaPOI code also contains legacy configurations that call
qwen-plus during forward multi-agent inference. Those configurations are not
the claimed deployed GeoSemID Predictor/Ranker protocol and must not be cited
as evidence for it.
