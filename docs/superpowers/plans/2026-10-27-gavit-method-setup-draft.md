# GAViT Methods and Experimental Setup Draft Implementation Plan

> **For agentic workers:** Execute this document-authoring plan in the current session. User selected LaTeX and the two English chapters explicitly. No additional writing-scope approval is required.

**Goal:** Produce an editable, compiled English partial dissertation containing methods and experimental setup, grounded in current code and archived evidence.

**Architecture:** Adapt the user-provided NTU template's typography and chapter structure in a new isolated document directory. Keep prose, bibliography and internal evidence notes separate. Do not change model code or infer final comparative results.

**Tech Stack:** LaTeX report, template preamble, IEEEtran BibTeX, Codex native compiler/editor.

**Spec:** `thesis/ntu_gavit/README.md`; user selections in this chat on 2026-10-27.

## Global Constraints

- English methods and experimental setup only; final abstract/results/conclusions are outside this iteration.
- Retain original dirty files and existing experiments; no commit, push, external email or GPU task.
- Fixed/soft grouping, 48+32 graph, GAT and feedback must match recorded implementation.
- Distinguish single-label calibration/refit from BEN validation-best testing and disclose historical comparability limits.
- Keep HGTNet/DFAGCN pending; do not fabricate results or significance claims.
- Independent factual review and actual compilation are required; no software test scaffolding for a text-only change.

## Review Focus

- Fixed pooling is learnable but fixed membership is not.
- Soft membership S and aggregation W normalise different axes.
- Cosine selects graph edges; GAT does not consume its returned edge_weight.
- Current 20%-pool NWPU must not be mixed with earlier70/15/15 results.
- Bibliography/labels must resolve; compilation alone does not verify scientific claims.

## Tasks

- [x] Audit current model/recipe sources, latest diary, core CSVs and archived supervisor requirements with two independent read-only agents.
- [x] Confirm user choices: LaTeX; English methods and experimental setup.
- [x] Create `thesis/ntu_gavit/main.tex`, two chapter sources, bibliography, template provenance, README and EVIDENCE notes.
- [x] Compile with the native LaTeX compiler and repair any errors.
- [x] Independently review chapter claims against authoritative files; correct material findings.
- [x] Validate citations/labels/source identity and final scope; record actual status in `validation.json` and research diary.
- [x] Request the editor project tab with live PDF preview; prepare source import package and concise delivery links. App returns queued; no claim that the tab has already been visually inspected.

## Execution rulings

- This is a document-only artifact. A dedicated directory isolates it without changing the user's branch/worktrees; the current branch is not main/master.
- The template's sample front matter and signatures are not copied into the review draft. Formal personal/declaration pages await full-manuscript assembly.
- No terminal TeX is present; the app's built-in compiler is the supported verification path. An exported PDF is optional, and no separate renderer is substituted for LaTeX compilation.
- Final native compile succeeded after caching initial format/dependency downloads and disabling XeTeX-incompatible font expansion. No scientific source correction was needed for the compile error.
- Numbered references and citation keys pass static checks; an independent manuscript review found no material method/protocol error. The two reporting clarifications and template numbering issue were corrected.
- Page-level visual QA was not completed: native compilation does not export a PDF, and computer-use access to Codex is unavailable. Deliver source and native-preview access without claiming a visually certified PDF.
