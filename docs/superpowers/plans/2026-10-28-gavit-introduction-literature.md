# GAViT Introduction and Literature Review Writing Plan

**Goal:** Add a substantive English introduction and literature review to the existing LaTeX dissertation project.

**Approved scope:** The user agreed to a 50–60-page dissertation including front matter and references, a working budget of 56 pages, and then explicitly approved writing Chapters 1 and 2. No additional planning approval is needed.

**Architecture:** Retain the template and existing Chapters 3–4. Author Chapter 1 around the problem, three research questions, bounded contributions and scope. Organise Chapter 2 by remote sensing representation learning, visual graph reasoning, region grouping and direct method comparisons, followed by a synthesis. Explain differences across methods instead of concatenating paper abstracts.

**Sources:** Original papers, publisher records and official project pages; existing Chinese literature draft is a starting point, not a substitute for checking citations. Keep exact metadata and a source-to-claim record. Aim to meet the supervisor's minimum of 40 relevant references across the current manuscript without padding unrelated work.

## Page budget agreed in chat

| Part | Target pages |
|---|---:|
| Front matter | 9 |
| Introduction | 4 |
| Literature review | 13 |
| Method | 8 |
| Experimental setup | 4 |
| Results and analysis | 8 |
| Discussion and limitations | 3 |
| Conclusions | 2 |
| References | 5 |
| Total | 56 |

The budget is provisional, not a measured page count. Target approximately 1,050–1,250 introduction words and 3,800–4,300 literature-review words; actual pagination also depends on tables and references.

## Tasks

- [x] Check dirty status, latest diary, early result CSVs and existing manuscript/provenance.
- [x] Audit original sources in parallel for remote sensing, visual graphs/grouping and general backbones.
- [x] Write `chapters/01-introduction.tex` and `chapters/02-literature-review.tex`; include an original comparison table.
- [x] Merge verified and actually cited BibTeX entries, deduplicating references; preserve the old eight-entry draft and import ZIP.
- [x] Update `main.tex` for Chapters 1–4, draft metadata, README, evidence notes and validation.
- [x] Independently review scientific claims, source attribution, status boundaries and chapter flow; retain the scope of actual reviewer feedback.
- [x] Run reference/label/input checks and native LaTeX compilation; repair source errors if any.
- [x] Refresh the versioned Overleaf import package and research diary, and deliver links.

## Constraints and review focus

- Do not modify experiment code, CSV scores or unrelated ongoing external-baseline work. No new training, evaluations, Git actions or external messages.
- Do not claim stable superiority, semantic segmentation, causal explanation or first-ever aggregation/feedback architecture.
- Swin already models contextual interactions. The question is whether the added regional inductive bias is useful.
- Distinguish node granularity, graph topology selection, learned aggregation, grouping membership and feedback.
- HGTNet/DFAGCN are direct related methods. Independent reproduction is being prepared; a completed common-protocol comparison is not yet established.
- Source existence/metadata checks do not mean every paper was read in full. Mark the actual evidence scope.
- Use seven-chapter organisation but only insert the four drafted chapters; do not fill unwritten results with invented text.
- Do not change font, margins or spacing to force a page count. References and front matter count toward the 50–60-page target.
- This is document authoring: use source/compile checks and independent academic review, not implementation-mirroring software tests.

## Execution record

- Parallel agents own disjoint new candidate/provenance files; the root agent alone merges the bibliography and integrates the manuscript.
- Retain pre-expansion main, bibliography, notes, validation and ZIP as a recoverable local snapshot.
- Native compilation confirms a build, not page-by-page visual QA. Existing inability to export/control the native PDF remains a reporting boundary unless a supported export becomes available.
- 2026-10-29: source estimates are1132 introduction words and3872 review words;50 independent references are all cited. Existing method/setup source hashes are unchanged.
- 2026-10-29: final native compiler returnedsuccess; citations, labels and relative inputs passed. This does not verify a13-page review or a50–60-page complete thesis.
- DGCNN volume38/issue5/article146 was corroborated by the coauthor institutional publication record already seen on10-28; do not import the placeholder1/1 pagination in the arXiv PDF. No unverified page range was added.

- 2026-10-29: resumed independent academic review completed with no important remaining issues reported; actual PDF pagination and visual QA remain pending.
