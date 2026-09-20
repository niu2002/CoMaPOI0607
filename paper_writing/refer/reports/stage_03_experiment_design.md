# Stage 03: Experiment Design and CoMaPOI-Inherited Protocol

## Design status

This paper is a GeoSemID extension of CoMaPOI. The following small protocol
subset is used as an inherited starting point, not as evidence for the new
method:

| Setting | Status | Source and treatment |
|---|---|---|
| Benchmarks: NYC, TKY, CA | ADAPTED_FROM_COMAPOI | Local CoMaPOI README and the verified CoMaPOI SIGIR 2025 record; all three are included in Mock tables |
| Last 30 check-ins per user for evaluation | ADAPTED_FROM_COMAPOI | Local CoMaPOI study summary; retain only after confirming the processed-data split script |
| 8B-class backbone with LoRA | ADAPTED_FROM_COMAPOI | Local README demonstrates Llama-3.1-8B LoRA commands; GeoSemID ranker backbone remains [TBD] |
| Candidate-size sensitivity | ADAPTED_FROM_COMAPOI | CoMaPOI reports a candidate-size study; GeoSemID evaluates its own K0/K grid with Mock values |
| CA / NYC / TKY statistics | MIXED | sample counts are verified locally; remaining statistics are [TBD] or cited as CoMaPOI protocol facts |

## Non-inheritance boundary

The GeoSemID paper does not inherit CoMaPOI's result tables, RRF contribution,
multi-agent ablations, candidate-recall values, or claims of superiority. RRF
remains an experimental comparison only; it is not a GeoSemID method component.
Every GeoSemID-specific table is explicitly marked Mock until all models are
rerun under a common protocol.

## Mock-design checks

- Three datasets: NYC/NYK, TKY, and CA.
- Separate candidate recall, conditional ranking, end-to-end ranking, oracle
  upper bound, ranker coverage, and router-supervision coverage.
- Ordinary semantic embedding is a competitive baseline.
- No p-values or claims of statistical significance.
- All new numeric values are isolated in experiments/mock_results.tex and
  labelled Mock in the manuscript.

## Skill output

paper-experiment-writing supplied the evidence boundary: inherited protocol
facts can structure the experiment, but only completed GeoSemID runs can
support performance conclusions.
