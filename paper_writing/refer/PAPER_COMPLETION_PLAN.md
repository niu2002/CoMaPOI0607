# GeoSemID T-ITS Completion Plan

## Scope and master source

- Target venue: IEEE Transactions on Intelligent Transportation Systems (T-ITS).
- English master source: ../english/GeoSemID_TITS_method.tex.
- Master-source status: latest locally modified GeoSemID IEEEtran manuscript; it
  compiles successfully before this completion work.
- Output root: this complete_mock directory. The original source is not
  overwritten.

## Frozen content

The following source interval is frozen:

- Start: \section{Problem Formulation}
- End: immediately before \bibliographystyle{IEEEtran}
- Snapshot: frozen_problem_method_snapshot.tex
- SHA-256: 8D6798B2EFCF5CB092DB2E5FB379CFCFBCE2B477EBF5A74032EAF7C4B6FE2C88

The frozen interval includes both Problem Formulation and Methodology, all
equations, labels, notation, GeoSemID, four retrieval experts, the
utility-supervised router, rank-percentile fusion, evidence preservation, and
candidate-constrained ranking. It must be copied unchanged into the completed
paper. Any deviation requires an itemized diff and justification.

## Current content and gaps

- Verified: Problem Formulation and Methodology; English abstract draft; Chinese
  method companion; CA datasets, candidate artifacts, and several CA evaluation
  logs.
- Missing: Introduction, Related Work, full experimental protocol, vetted
  BibTeX, Conclusion, complete CA/NYK/TKY result matrices, final hyperparameters,
  and publication-ready reproducibility facts.
- Mock policy: all unverified numerical values are isolated in
  experiments/mock_results.tex and visibly marked as non-empirical layout
  values. Existing CA logs are retained as audit evidence, not overwritten.

## Local evidence to use

- dataset_all/ca_train.jsonl and dataset_all/ca_test.jsonl: 6,616 and 1,818
  samples.
- dataset_all/nyc_train.jsonl and dataset_all/nyc_test.jsonl: 3,870 and 988
  samples. The manuscript uses NYK only after author confirmation of the
  dataset naming convention.
- dataset_all/tky_train.jsonl and dataset_all/tky_test.jsonl: 11,850 and
  2,206 samples.
- The local CoMaPOI material supplies a reference experimental protocol:
  the repository supports NYC, TKY, and CA, and code constants list 5,091,
  7,851, and 13,630 maximum item identifiers, respectively. These are
  implementation references, not substitute manuscript dataset statistics.
- dataset_all/ca/ca_poi_info.csv: 13,628 POIs.
- results/ca/*/metrics.txt: partial CA metrics with different run sizes;
  unsuitable for a final common-protocol comparison.
- Retrieval and candidate code: rag/RAG.py, rag/expertrag.py, and
  candidate_fusion.py.

## CoMaPOI inheritance rule

GeoSemID extends the local CoMaPOI project. The completed experimental setting
may therefore inherit a small, clearly attributed protocol subset from the
published CoMaPOI study: the NYC/TKY/CA benchmark family, last-30-check-in
evaluation convention, and the use of a LoRA-adapted 8B-class backbone as a
reference configuration. Inherited facts are recorded as
ADAPTED_FROM_COMAPOI and cited to the verified CoMaPOI SIGIR 2025 paper. They
are never presented as a verified GeoSemID result or configuration. All
GeoSemID-specific settings remain [TBD] until confirmed by implementation.

## Literature plan

1. Query DBLP for representative sequential, spatiotemporal, graph, semantic
   identifier, two-stage retrieval, dynamic fusion, and LLM-POI work.
2. Admit a reference only after the title, authors, venue, year, bibliographic
   metadata, canonical DBLP record, and BibTeX key are verified.
3. Record every admitted paper in literature/references_verified.csv, then
   synthesize it in literature/literature_map.md.

## Page budget

- Abstract: 0.2 page.
- Introduction and Related Work: at most 1.5 IEEE two-column pages.
- Frozen Problem Formulation and Methodology: unchanged.
- Experiments: approximately four pages, with visibly marked Mock tables.
- Conclusion: 0.2--0.3 page plus references.

## Stages and acceptance criteria

1. Project audit: inventory local sources, code, datasets, logs, and frozen
   hash; classify facts as verified, inferred, missing, or author-confirmed.
2. Literature research: DBLP-verifiable sources only; no guessed metadata.
3. Experiment design: internally consistent Mock tables, explicit [TBD]
   facts, and a real-result replacement guide.
4. Writing: Introduction, exactly two Related Work subsections, Experiments,
   and Conclusion, each with conservative claims.
5. Integration: complete IEEEtran source, BibTeX compilation, page count,
   undefined-reference/overfull checks, and frozen-hash comparison.
6. Review: technical-closure and T-ITS reviewer reports, Chinese review copy,
   and author-input checklist.

## Final deliverables

GeoSemID_TITS_complete_mock.tex, PDF, references.bib, frozen snapshot, four
section files, Mock result files, literature ledger/map, stage reports,
reference and reviewer audits, Chinese review copy, and
AUTHOR_INPUTS_REQUIRED.md.

## Author confirmation still required

Exact Qwen/generator and retrieval models, semantic-label provenance and
quality control, GeoSemID vocabularies and geographic resolutions, complete
dataset statistics, final split protocol, $K_0$, $K$, router/ranker
hyperparameters, hardware, random seeds, statistical testing, availability
links, and verified final results.
