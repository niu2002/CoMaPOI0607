from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
HOME = Path.home()
GLOBAL_SKILLS = HOME / ".agents" / "skills"
REPO_SKILLS = ROOT / ".agents" / "skills"
GLOBAL_AGENTS = HOME / ".codex" / "AGENTS.md"


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8", newline="\n")


def upsert_tail_section(path: Path, marker: str, block: str) -> None:
    existing = path.read_text(encoding="utf-8") if path.exists() else ""
    if marker in existing:
        existing = existing[: existing.index(marker)].rstrip() + "\n\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    sep = "\n\n" if existing and not existing.endswith("\n\n") else ""
    path.write_text(existing + sep + block.strip() + "\n", encoding="utf-8", newline="\n")


def yaml_meta(display_name: str, short_description: str, default_prompt: str) -> str:
    return f"""
interface:
  display_name: "{display_name}"
  short_description: "{short_description}"
  default_prompt: "{default_prompt}"
"""


SKILLS = {
    "paper-deep-research": {
        "display": "Paper Deep Research",
        "short": "Research a paper topic with verifiable sources before writing.",
        "prompt": "Use $paper-deep-research to investigate this paper topic and list missing inputs before drafting.",
        "body": r"""
---
name: paper-deep-research
description: Use when Codex needs to investigate a paper topic, survey recent or foundational literature, collect sources, find research gaps, or prepare evidence for Related Work, Introduction, Background, or thesis writing. Trigger for requests mentioning deep research, 文献调研, related work, literature survey, research gap, citation candidates, or using dzhng/deep-research for academic writing.
---

# Paper Deep Research

Use this skill to build an evidence base before drafting academic text.

## Source policy

- Prefer primary sources: papers, official proceedings, arXiv pages, DOI pages, project pages, datasets, and official documentation.
- Use Zotero or an existing `.bib` file when available; otherwise return citation candidates marked for verification.
- Do not invent citations, venues, years, metrics, ablations, or limitations.
- Mark unsupported claims as `[CITATION NEEDED]`.
- For fast-moving topics, browse or use a configured research tool before answering.

## Workflow

1. Restate the research question in one sentence.
2. List missing inputs before running research: target venue, year range, required citation style, seed papers, exclusion criteria, and whether Chinese/English output is needed.
3. If `dzhng/deep-research` is installed or available, use it as the evidence collector:
   - Set a focused query rather than a broad paper title dump.
   - Use low breadth/depth for quick checks and higher breadth/depth for survey sections.
   - Save the generated report path and cite it as intermediate evidence, not as a final scholarly source.
4. If `deep-research` is unavailable, use web search, Zotero, local PDFs, or BibTeX directly.
5. Produce a research brief with:
   - Key clusters of work.
   - Representative papers with source links or BibTeX keys.
   - Claims supported by each cluster.
   - Open gaps and how the user's paper can position itself.
   - A list of unresolved facts that still need user confirmation.

## `deep-research` integration notes

`dzhng/deep-research` is a Node/TypeScript research agent that uses Firecrawl search/scrape and an LLM to recursively generate learnings and a final Markdown report. It is useful as a first-pass collector, but its output must be checked against original sources before appearing in a paper.

If the user asks to install or run it, check for:

- Node.js and package manager availability.
- `FIRECRAWL_KEY` or equivalent Firecrawl configuration.
- LLM API key and model configuration.
- A target output directory inside the workspace.

Ask for missing credentials or install approval when needed.
""",
    },
    "paper-literature-synthesis": {
        "display": "Paper Literature Synthesis",
        "short": "Turn papers and notes into a structured Related Work.",
        "prompt": "Use $paper-literature-synthesis to organize these papers into a Related Work section.",
        "body": r"""
---
name: paper-literature-synthesis
description: Use when Codex needs to synthesize papers into Related Work, Background, literature review, taxonomy, comparison tables, research gaps, or citation-backed academic paragraphs. Trigger for 文献综述, 相关工作, literature synthesis, taxonomy, survey table, compare methods, or positioning a paper against prior work.
---

# Paper Literature Synthesis

Use this skill after evidence has been collected from PDFs, Zotero, BibTeX, notes, or `paper-deep-research`.

## Required inputs

Before drafting, identify what is available:

- Target section: Introduction, Related Work, Background, Method comparison, or Discussion.
- Paper topic and claimed contribution.
- Citation source: BibTeX keys, Zotero items, PDFs, DOI links, or source URLs.
- Target venue style if known.

If citation material is missing, produce an outline and mark citations as `[CITATION NEEDED]`.

## Synthesis workflow

1. Group papers by technical idea, dataset/task, modeling paradigm, or limitation.
2. For each group, extract: what problem it addresses, what method family it uses, why it matters, and what limitation remains.
3. Build a comparison matrix when there are at least four papers.
4. Write from broad to narrow:
   - general problem landscape,
   - dominant method families,
   - specific gap relevant to the user's work,
   - how the current paper differs.
5. Keep claims conservative and source-tied.

## Output shape

Return:

- A short taxonomy.
- A draft section in academic English or Chinese as requested.
- A citation TODO list for every unsupported claim.
- A list of papers that are likely missing from the current source set.

## Quality checks

- Do not summarize papers one by one unless the user explicitly asks for annotated bibliography.
- Avoid exaggerated claims such as "first", "novel", or "significant" unless the evidence supports them.
- Keep method names, dataset names, metrics, and model acronyms exact.
""",
    },
    "paper-academic-writing": {
        "display": "Paper Academic Writing",
        "short": "Draft, translate, polish, shorten, or expand academic prose.",
        "prompt": "Use $paper-academic-writing to revise this paper text while preserving meaning and citations.",
        "body": r"""
---
name: paper-academic-writing
description: Use when Codex needs to draft, translate, polish, shorten, expand, de-AI, or restructure academic paper text, including titles, abstracts, introductions, method sections, conclusions, cover letters, and rebuttals. Trigger for 论文润色, 学术翻译, polish, rewrite, reduce word count, improve logic, make it more academic, or improve English/Chinese paper writing.
---

# Paper Academic Writing

Use this skill for paper prose. Preserve technical meaning, citations, variables, formulas, dataset names, and numerical results.

## First checks

- Identify the target language, section type, and desired tone.
- Ask for target venue, word limit, and LaTeX constraints if they matter.
- If the text includes Chinese, follow the global UTF-8 safety rule for any file edits.
- If the user asks for rewriting a file, inspect the surrounding section before editing.

## Revision modes

Choose the smallest mode that matches the request:

- **Polish:** improve grammar, flow, concision, and academic tone without changing claims.
- **Translate:** translate faithfully, preserving technical terms and citation commands.
- **Logic edit:** reorder sentences, expose missing transitions, and remove repeated ideas.
- **Shorten:** reduce length while preserving core claims, evidence, and citations.
- **Expand:** add motivation, transitions, or explanation only when the needed facts are present.
- **Rebuttal:** be factual, respectful, and directly tied to reviewer comments.

## Output rules

- Preserve LaTeX commands, labels, refs, citations, math, and table/figure references.
- Do not fabricate citations, results, or limitations.
- Use `[NEEDS DATA]`, `[CITATION NEEDED]`, or `[USER CONFIRM]` when information is missing.
- For substantial edits, provide a concise change summary after the revised text.
- For file edits, validate UTF-8 and parse/build when practical.

## Style preferences

- Prefer clear, direct academic prose over ornate wording.
- Avoid hype words unless supported: groundbreaking, unprecedented, significant, remarkable.
- Keep model names, method names, datasets, acronyms, and metrics in English unless the user requests otherwise.
""",
    },
    "paper-experiment-writing": {
        "display": "Paper Experiment Writing",
        "short": "Write results, tables, figures, ablations, and error analysis from real data.",
        "prompt": "Use $paper-experiment-writing to turn these experiment results into paper-ready analysis.",
        "body": r"""
---
name: paper-experiment-writing
description: Use when Codex needs to write or check experiment sections, result analysis, ablation studies, tables, figure captions, metric explanations, error analysis, or reproducibility text for an academic paper. Trigger for 实验分析, ablation, results, table caption, figure caption, metrics, reproducibility, baseline comparison, or interpreting model performance.
---

# Paper Experiment Writing

Use this skill to turn real experimental artifacts into defensible paper text.

## Evidence requirements

Before writing conclusions, locate the actual evidence:

- Tables, logs, CSV/XLSX files, JSON outputs, screenshots, or notebooks.
- Dataset split, metric definitions, baselines, and statistical settings.
- Whether higher or lower is better for each metric.

If evidence is missing, write an analysis scaffold and mark claims as `[NEEDS DATA]`.

## Workflow

1. Extract the key numerical comparisons.
2. Check whether every claim follows from the numbers.
3. Separate observations from explanations:
   - Observation: what the table shows.
   - Interpretation: why it may happen.
   - Limitation: what the result does not prove.
4. Write concise paper-ready paragraphs for:
   - Main results.
   - Ablation study.
   - Sensitivity analysis.
   - Efficiency/runtime.
   - Error cases.
5. Create captions that explain what is measured and why the reader should care.

## Guardrails

- Do not claim statistical significance unless tests were run.
- Do not call a result "best" unless all compared methods and metrics support it.
- Do not hide negative results; frame them as limitations or future work.
- Keep all numbers, units, and metric names exact.
""",
    },
    "paper-reviewer-check": {
        "display": "Paper Reviewer Check",
        "short": "Review a draft like a strict academic reviewer.",
        "prompt": "Use $paper-reviewer-check to review this draft and list fixable weaknesses.",
        "body": r"""
---
name: paper-reviewer-check
description: Use when Codex needs to review an academic paper draft, thesis chapter, rebuttal, experiment section, or submission package for reviewer-facing weaknesses, missing citations, unsupported claims, contribution clarity, reproducibility, formatting risks, or acceptance readiness. Trigger for reviewer check, 审稿人视角, paper review, submission checklist, rebuttal risk, or camera-ready check.
---

# Paper Reviewer Check

Use this skill as a critical pre-submission pass.

## Review stance

Prioritize issues that could affect acceptance:

- unclear problem or contribution,
- unsupported novelty,
- missing or unfair baselines,
- weak ablations,
- unverified citations,
- inconsistent notation,
- dataset or metric ambiguity,
- claims not supported by experiments,
- reproducibility gaps,
- venue formatting or anonymity risks.

## Workflow

1. Identify the target venue or thesis context.
2. Inspect the relevant draft files, figures, tables, and bibliography when available.
3. List findings by severity:
   - Critical: likely rejection or correctness risk.
   - Major: weakens evidence or clarity.
   - Minor: style, grammar, or formatting.
4. For each finding, include the exact file/section/line if available, why it matters, and a concrete fix.
5. End with missing information needed from the user.

## Output rules

- Findings first, summary second.
- Be direct but constructive.
- Do not rewrite the whole paper unless asked.
- Do not invent reviewer objections unsupported by the draft.
""",
    },
}


COMAPOI_SKILL = r"""
---
name: comapoi-paper-writer
description: Use for writing, revising, or reviewing the CoMaPOI/JJY paper in this repository, especially POI recommendation/prediction, geospatial error, Haversine Distance, result tables, LaTeX paper text, Chinese-English academic writing, and consistency between code, experiments, figures, and the manuscript.
---

# CoMaPOI Paper Writer

Use this repository-specific skill when working on the JJY CoMaPOI paper.

## Project context

- The repository contains JJY's paper code and manuscript materials.
- Existing skill `comapoi_prediction_analyzer` handles multidimensional academic metrics, spatial physical geographic error, Haversine Distance, and paper-grade figures.
- Use this skill for writing and consistency work around those analyses.

## Workflow

1. Locate the manuscript, experiment outputs, figures, tables, and bibliography before editing claims.
2. Use `comapoi_prediction_analyzer` when the task needs metric computation, Haversine Distance analysis, spatial error analysis, or academic figures.
3. Keep POI names, dataset names, metrics, method names, and model names exact.
4. For Chinese-bearing files, follow the global UTF-8 safety rule before and after edits.
5. When drafting paper text:
   - connect claims to actual experiment outputs,
   - mark missing citations as `[CITATION NEEDED]`,
   - mark missing result evidence as `[NEEDS DATA]`,
   - avoid claiming novelty or superiority unless the manuscript evidence supports it.
6. When reviewing:
   - check consistency between abstract, introduction, method, experiments, captions, and conclusion,
   - check whether every table/figure is referenced and interpreted,
   - check whether geospatial claims match the reported Haversine or distance-error results.

## Preferred outputs

- For writing tasks: revised paper text plus a short list of changed claims.
- For review tasks: severity-ranked findings with exact file/section references.
- For missing context: a compact checklist of the files or facts needed from the user.
"""


GLOBAL_PROTOCOL = r"""
## Academic Paper Workflow

When I ask Codex to help write, revise, review, translate, or research a paper:

1. First identify the task type: deep research, literature synthesis, prose editing, experiment writing, or reviewer check.
2. Use the matching paper skill from `~/.agents/skills` when available.
3. Ask for missing scholarly inputs only when they are required: target venue, draft path, bibliography, source PDFs, experiment outputs, figure/table files, word limit, or desired language.
4. Never invent citations, venues, years, datasets, metrics, baselines, numerical results, or experimental conclusions.
5. Mark unsupported claims as `[CITATION NEEDED]`, missing experimental evidence as `[NEEDS DATA]`, and decisions requiring me as `[USER CONFIRM]`.
6. Prefer Zotero, BibTeX, local PDFs, official proceedings, DOI/arXiv pages, and primary sources over model memory.
7. For Chinese-bearing paper files, follow the 中文 UTF-8 Safety Hard Rule above before and after edits.
8. For this repository, use `comapoi-paper-writer` together with `comapoi_prediction_analyzer` when writing CoMaPOI experiment or spatial-error analysis.
"""


def install() -> None:
    for name, data in SKILLS.items():
        folder = GLOBAL_SKILLS / name
        write_text(folder / "SKILL.md", data["body"])
        write_text(
            folder / "agents" / "openai.yaml",
            yaml_meta(data["display"], data["short"], data["prompt"]),
        )

    write_text(REPO_SKILLS / "comapoi-paper-writer" / "SKILL.md", COMAPOI_SKILL)
    write_text(
        REPO_SKILLS / "comapoi-paper-writer" / "agents" / "openai.yaml",
        yaml_meta(
            "CoMaPOI Paper Writer",
            "Write and review the CoMaPOI paper using repository evidence.",
            "Use $comapoi-paper-writer to improve this CoMaPOI paper section.",
        ),
    )
    upsert_tail_section(GLOBAL_AGENTS, "## Academic Paper Workflow", GLOBAL_PROTOCOL)


def validate() -> None:
    targets = [GLOBAL_AGENTS]
    targets.extend((GLOBAL_SKILLS / name / "SKILL.md") for name in SKILLS)
    targets.extend((GLOBAL_SKILLS / name / "agents" / "openai.yaml") for name in SKILLS)
    targets.append(REPO_SKILLS / "comapoi-paper-writer" / "SKILL.md")
    targets.append(REPO_SKILLS / "comapoi-paper-writer" / "agents" / "openai.yaml")

    expected_phrases = ["文献调研", "论文润色", "实验分析", "审稿人视角", "中文 UTF-8"]
    problems = []
    for path in targets:
        text = path.read_text(encoding="utf-8")
        q_count = text.count("?")
        repl_count = text.count("\ufffd")
        if repl_count:
            problems.append(f"{path}: replacement chars={repl_count}")
        if path.name == "SKILL.md":
            if not re.match(r"^---\nname: [a-z0-9-]+\ndescription: .+\n---\n", text, re.S):
                problems.append(f"{path}: invalid frontmatter")
        if path.suffix in {".yaml", ".yml"} and "interface:" not in text:
            problems.append(f"{path}: missing interface")
        print(f"{path} | question_marks={q_count} | replacement_chars={repl_count}")

    combined = "\n".join(path.read_text(encoding="utf-8") for path in targets)
    for phrase in expected_phrases:
        print(f"expected_phrase[{phrase}]={phrase in combined}")
        if phrase not in combined:
            problems.append(f"missing expected phrase: {phrase}")

    if problems:
        raise SystemExit("\n".join(problems))


if __name__ == "__main__":
    install()
    validate()
    print("paper workflow installed")
