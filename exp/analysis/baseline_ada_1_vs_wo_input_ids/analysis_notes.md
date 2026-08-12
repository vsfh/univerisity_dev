# Analysis notes

## Report structure mapping

- Title: `title`
- Technical summary: `technical-summary`
- Key findings with visual evidence: reported loss, pure image loss, alignment loss, component table, historical evaluation
- Scope/data/definitions: `scope`
- Methodology and experiment specification: `method`
- Limitations/robustness: `limitations`
- Recommended next steps: `next-steps`
- Further questions: `questions`

## Chart map

| Section | Question | Family/type | Fields | Supported claim |
|---|---|---|---|---|
| Reported total loss | Why does the latest text run remain higher? | Trend / line | epoch, reported_train_loss, run | The gap is expected while the text term stays active. |
| Pure image retrieval | Did the image branch converge differently? | Trend / line | epoch, pure_image_retrieval_loss, run | Curves nearly overlap after loss decomposition. |
| Text alignment | Does fixed text supervision learn stably? | Trend / line + reference | epoch, text_alignment_loss | Alignment CE falls steadily below the uniform baseline. |

Repeated line charts are intentional because all three questions concern continuous 20-epoch optimization paths; the second is a decomposition validation of the first, and the third is a distinct auxiliary objective. Exact final values and historical evaluation use tables rather than additional charts.

## Source and QA notes

- The latest complete event is selected for the current `baseline_ada_1_5x3` checkpoint.
- The earlier complete event is retained and explicitly labelled historical because its evaluation JSON exists.
- The 88-byte empty event is omitted.
- The latest of four complete no-text events is selected because it matches the current checkpoint timestamp; three earlier reruns are omitted.
- Latest fixed-weight model evaluation is omitted because no matching result exists.
- Historical clustered confidence intervals are descriptive robustness checks; no causal claim is made.

### Raw source inventory

- Current fixed text event: `exp/runs/baseline_ada_1_5x3/events.out.tfevents.1786420717.4090-48g.511988.0`
- Historical scheduled text event: `exp/runs/baseline_ada_1_5x3/events.out.tfevents.1786381215.4090-48g.4013619.0`
- Current no-text event: `exp/runs/baseline_wo_input_ids_ada_5x3/events.out.tfevents.1786302112.4090-48g.2760507.0`
- Historical text evaluation: `exp/eval_results/baseline_ada_1_5x3.json`
- No-text evaluation: `exp/eval_results/baseline_wo_input_ids_ada_5x3.json`
- Effective configs: `exp/outputs/baseline_ada_1_5x3/effective_config.json`, `exp/outputs/baseline_wo_input_ids_ada_5x3/effective_config.json`
