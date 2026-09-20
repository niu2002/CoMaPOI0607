# Mock-result replacement guide

All values in mock_results.tex and mock_results.csv are manuscript-layout
values, not empirical findings. Replace a macro only after its corresponding
common-protocol run is archived with dataset, split, seed, checkpoint, and
metric script information. Preserve the following separations:

1. candidate Recall@K;
2. conditional ranking metrics on retrieved targets;
3. end-to-end metrics with retrieval misses counted as errors;
4. oracle-candidate diagnostic upper bound;
5. ranker-training and router-supervision coverage.

For each replacement, verify monotonic Recall with candidate size, end-to-end
HR no greater than candidate recall, oracle no lower than end-to-end, and
consistent mean/std reporting. Do not insert p-values unless a defined test has
been run.
