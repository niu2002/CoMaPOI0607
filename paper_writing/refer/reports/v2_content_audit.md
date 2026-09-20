# V2 Content Audit

## Frozen content

Problem Formulation and Methodology are included through
frozen_problem_method_snapshot.tex. V2 SHA-256 is
8D6798B2EFCF5CB092DB2E5FB379CFCFBCE2B477EBF5A74032EAF7C4B6FE2C88.
No V2 change is authorized within that snapshot.

## Abstract

The V1 abstract has a placeholder result and conflates the GeoSemID identifier
with the full framework. It does not state the Qwen-Plus teacher to local
Predictor training path. V2 will name GeoSemID as the representation module,
retain result macros, and distinguish offline generator, teacher, and student.

## Introduction and Related Work

V1 has a coherent problem chain but too few citations for its claims. Related
Work does not yet adequately cover semantic identifiers, discrete codes,
retrieve-then-rank, routing, teacher-generated supervision, or LLM POI work.
V2 expands only with references that can be verified; unresolved coverage is
recorded in v2_reference_audit.md rather than filled with guessed citations.

## Experiments

V1 contains only two small Mock tables. V2 replaces this with a unified Mock
data source, consistency checker, coverage/error decomposition, representation,
routing, evidence, teacher-student, long-tail, and cost analyses. Every
generated number is marked MOCK -- NOT EMPIRICAL CLAIMS.

## Method--implementation boundary

The documented target protocol is: local offline GeoSemID generator; Qwen-Plus
teacher for reverse-reasoning supervision; fine-tuned local Predictor/Ranker
for test-time inference. Existing legacy CoMaPOI cloud-forward configurations
are not cited as the V2 deployed protocol.
