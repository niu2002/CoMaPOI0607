# Stage 01: Project and Evidence Audit

## Master manuscript

| Item | Status | Evidence |
|---|---|---|
| English GeoSemID IEEEtran source | VERIFIED_FROM_FILE | ../english/GeoSemID_TITS_method.tex |
| Compilable method-only PDF | VERIFIED_FROM_FILE | ../english/GeoSemID_TITS_method.pdf |
| Frozen Problem Formulation + Methodology | VERIFIED_FROM_FILE | frozen_problem_method_snapshot.tex; SHA-256 recorded in PAPER_COMPLETION_PLAN.md |
| Prior Chinese method companion | VERIFIED_FROM_FILE | ../chinese/GeoSemID_TITS_method_zh.tex |

## Dataset and experiment materials

| Item | Status | Evidence |
|---|---|---|
| CA train/test JSONL | VERIFIED_FROM_FILE | 6,616 / 1,818 lines in dataset_all/ca_train.jsonl and ca_test.jsonl |
| NYC train/test JSONL | VERIFIED_FROM_FILE | 3,870 / 988 lines in dataset_all/nyc_train.jsonl and nyc_test.jsonl |
| TKY train/test JSONL | VERIFIED_FROM_FILE | 11,850 / 2,206 lines in dataset_all/tky_train.jsonl and tky_test.jsonl |
| CA POI catalog | NEEDS_AUTHOR_CONFIRMATION | 13,628 local metadata rows conflict with an earlier summary value of 13,564 processed POIs |
| Formal dataset names | VERIFIED_FROM_FILE | code, README, and data directories consistently use NYC, TKY, and CA; NYK is not a local identifier |
| CA partial metrics | VERIFIED_FROM_FILE | multiple results/ca/*/metrics.txt files, but run sizes differ |
| Comparable final baseline matrix | MISSING | no common split/protocol table found |
| Spatial error diagnostic PDF | VERIFIED_FROM_FILE | CA interim diagnostic PDF in results/ca/amd_rag_lora_with_sft_ca/ |

## Implementation evidence

| Component | Status | Evidence |
|---|---|---|
| POI/trajectory embedding retrieval | VERIFIED_FROM_FILE | rag/RAG.py encodes POI descriptions and queries; supports local/API embeddings |
| GeoSemID/HSID text augmentation | VERIFIED_FROM_FILE | rag/RAG.py appends hsid_text when enabled |
| Spatial and semantic candidate retrieval | VERIFIED_FROM_FILE | rag/expertrag.py |
| Heuristic four-expert fusion | INFERRED_FROM_CODE | current expertrag.py contains a legacy RRF-style implementation |
| Utility-supervised router | MISSING | frozen Method defines the intended paper method; no matching current training implementation was confirmed |
| Candidate-constrained likelihood ranker | PARTIALLY_INFERRED_FROM_CODE | local inference/fine-tuning scripts exist, but the frozen protocol requires final implementation confirmation |
| Prompt serialization | VERIFIED_FROM_FILE | prompt_provider.py and rag/RAG.py |
| CoMaPOI experimental setting reference | VERIFIED_FROM_FILE | README.md, Forward_Inference.md, and local configuration constants support NYC, TKY, and CA; use only values confirmed by files |

## Audit decision

Only verified local CA statistics and partial diagnostics may be mentioned as
existing artifacts. The completed paper therefore uses explicitly marked Mock
tables for the planned common-protocol comparison. No Mock value replaces or
relabels an existing real result.

## Skill outputs in this stage

- comapoi-paper-writer: established consistency scope between GeoSemID,
  retrieval, ranking, and available CA artifacts.
- paper-reviewer-check: identified the absence of a common-protocol baseline
  matrix and router implementation evidence as publication blockers, not facts
  to be filled with invented numbers.
