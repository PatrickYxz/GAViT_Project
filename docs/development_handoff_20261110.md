# Development handoff — 2026-11-10

This snapshot preserves two independent development branches. They have not been merged.

## Current thesis, external baselines and new campaign

Branch: `codex/token-ffn-nwpu-20261003`.

```sh
git clone --branch codex/token-ffn-nwpu-20261003 https://github.com/PatrickYxz/GAViT_Project.git
```

For an existing clean clone:

```sh
git fetch origin
git switch codex/token-ffn-nwpu-20261003
git pull --ff-only
```

Start with `research_diary.md`, `docs/advisor_work_schedule_20261109.md`, and `results/evidence/campaign_prepare_20261110/frozen_campaign.json`.

- Thesis source: `thesis/ntu_gavit/`; latest verified partial PDF: `output/pdf/ntu_gavit_external_results_20261108.pdf`; Overleaf archive: `thesis/ntu_gavit_overleaf_20261108.zip`.
- Preserve all favourable and unfavourable historical seeds. Missing seed46/47 original records and F1 remain missing; do not reconstruct them as measurements.
- Current campaign: nine new formal runs, one GPU at a time. Order: AID soft smoke, soft42, DFAGCN43/44/45, AID soft43/44/45, Swin45, fixed45. The smoke is an additional bounded engineering check, not a formal result.
- Prepared bundles: `output/aid_campaign_20261110_v1.pyz` and `output/dfagcn_campaign_20261110_v1.pyz`. The original bundles retain their historical uncommitted-source snapshot identities; committing them does not retroactively change their embedded Git provenance.
- Fresh local checks before this handoff: AID campaign 26, DFAGCN campaign 13, and AID result analysis 4 tests passed. These are local targeted checks, not full-suite, CUDA or training evidence. The local environment lacks the ML dependencies needed for the full tensor suite.
- A durable sequential supervisor was discussed but has not been implemented or launched. Do not assume an automatic queue or heartbeat is running.

## Historical sparse-hybrid experiments and soft-assignment development

Branch: `codex/sparse-hybrid-4n-top2`.

Use a separate clone or worktree to inspect this branch; do not overwrite a dirty checkout:

```sh
git clone --branch codex/sparse-hybrid-4n-top2 https://github.com/PatrickYxz/GAViT_Project.git GAViT_sparse_hybrid
```

This branch contains its own research diary, BigEarthNet comparison history, soft-assignment/model changes, training and evaluation entry points, scene-classification code, diagnostic tooling and experimental evidence. It is preserved independently rather than silently merged into the thesis/campaign branch.

Static compilation checked 777 Python files in the current branch and 177 in this worktree without syntax failures. The sparse worktree's grouping-identity check ran four cases successfully, but its metadata case could not import `utils` because NumPy is absent; the scene-data test module could not import Pillow. These are unresolved local dependency gates, not a passing full tensor suite. This push is a development snapshot rather than a release or a declaration that every experiment is runnable locally.

## GPU and SSH state at handoff

The last successful read-only query used `workspace.featurize.cn:27435`, user `featurize`, and public-key authentication. It returned RTX3080, UUID `GPU-0d9901ca-d2e0-d398-b9b4-2550e6db2fed`, 10240 MiB total / 9874 MiB free, driver 610.57.04. This is an observation, not a guarantee that the same instance will still exist on the next machine.

Mac hotspot address `172.20.10.9` with the global proxy left enabled allowed one successful SSH query using a physical-interface connection. Further reconnections were intermittent. An authenticated shell was opened, but no training commands were submitted; the shell has since closed. No smoke, formal training, data recovery or automatic queue was started in this session.

The successful endpoint-scoped route used `nc -b en0 -G 12 -w 60 183.223.25.19 27435`. It is macOS-specific and must be adapted and reverified on another computer; do not reuse retired ports15437/18307. Server host keys must be verified normally for the new machine.

Private keys, passwords, local SSH config/known_hosts, virtual environments, datasets/checkpoints and temporary downloads are not transferred through Git. Configure the new machine's own key through the user's normal Featurize account flow. Raw images belong on instance-local storage, while code, logs, weights and checkpoints remain in persistent `work`.

Before the next launch, read `docs/featurize_runbook.md`, inventory GPU/processes/software/disk and persistent inputs, then pass the actual real-data CUDA smoke. Preserve any failure and stop the queue rather than automatically changing the frozen recipe.
