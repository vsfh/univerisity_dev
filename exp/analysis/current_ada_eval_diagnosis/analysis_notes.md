# Analysis notes

## Required structure mapping

- Title: `title`
- Technical summary: `technical-summary`
- Key findings with visual evidence: provenance, retrieval, localization, center distribution, mechanism, and training components
- Scope/data/metric definitions: `definitions`
- Methodology: embedded in `definitions` and source query metadata
- Limitations/uncertainty/robustness: `limitations`
- Recommended next steps: `next-steps`
- Further questions: `questions`

## Chart map

| Section | Question | Family/type | Fields | Supported claim |
|---|---|---|---|---|
| Retrieval | How large is each rank-cutoff regression? | Comparison / bar | metric, delta_pp | R@1/5/10 each decline about 0.7pp. |
| Localization | Which overlap metrics move most? | Comparison / bar | metric, delta_points | IoU threshold and mean metrics consistently decline. |
| Slices | Broad or concentrated regression? | Relationship / scatter | r1_delta_pp, miou_delta_points, height | All 32 mIoU slice deltas are negative. |
| Center error | Why does mean CDE improve? | Distribution / threshold-CDF line | threshold, share, model | Ada loses near-center precision but improves the extreme tail. |

Palette policy is a hard two-root cap plus neutrals for focal-vs-baseline comparisons. The subset scatter uses four height categories because height identity is analytically relevant. Zero reference lines carry sign without relying on color.

## Source and QA notes

- Current Ada JSON mtime: 2026-08-11 18:27:27 +0800; checkpoint mtime: 17:11:48.
- Current Ada checkpoint SHA256 observed during this audit: `ca4cfb62053d8958a295928e8622e12d04db15074738f70b539c3b91296a3be4`.
- JSON files do not persist hashes, seeds, include-map hashes, or code commits; chronology and exact path matching provide high-confidence but not cryptographic run attribution.
- Query keys, population, candidate size, gallery size, satellite dimensions, crop ratio, and deterministic candidate seed match exactly.
- Cluster bootstrap intervals cover test-query sampling by gallery label, not training-run seed variance.
- Caption collision probabilities are expected values under random shuffle; actual batch satellite IDs were not logged.
- The previous training-curve report was generated before the current Ada JSON overwrote the old scheduled-run JSON and is superseded for current evaluation conclusions.

### Raw source inventory

- Current fixed-text evaluation: `exp/eval_results/baseline_ada_1_5x3.json`
- Current no-text evaluation: `exp/eval_results/baseline_wo_input_ids_ada_5x3.json`
- Current fixed-text event: `exp/runs/baseline_ada_1_5x3/events.out.tfevents.1786420717.4090-48g.511988.0`
- Current no-text event: `exp/runs/baseline_wo_input_ids_ada_5x3/events.out.tfevents.1786302112.4090-48g.2760507.0`
- Code paths: `exp/train.py`, `exp/train_ada.py`, `exp/model.py`, `exp/dataset.py`, `exp/test.py`
- Current effective configs: `exp/outputs/baseline_ada_1_5x3/effective_config.json`, `exp/outputs/baseline_wo_input_ids_ada_5x3/effective_config.json`
- Historical scheduled-text snapshot: `exp/analysis/baseline_ada_1_vs_wo_input_ids/analysis.sqlite`
- Caption corpus profiled from the 701 train IDs and per-satellite `qwen_6_28_description.json` files.
