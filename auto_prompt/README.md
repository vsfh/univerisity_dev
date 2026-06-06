# Auto Prompt Research

This folder runs prompt-only research for the Qwen description generator and the trained `Encoder_test` retrieval network.

Default single iteration:

```bash
python auto_prompt/run_iteration.py --iteration 1 --overwrite-descriptions
```

What it does:

- reads `auto_prompt/current_prompt.md`
- uses `tools/generate_qwen_6_4.py` through its existing public functions, without modifying that file
- generates descriptions for the first 50 test cases into `auto_prompt/generated_descriptions/`
- loads `/data/feihong/ckpt/model_test_geo_input_ids/last.pth`
- evaluates retrieval with `Encoder_test`
- appends metrics and the full prompt to `auto_prompt/result.csv`
- writes full prompt history to `auto_prompt/prompt_history.jsonl`
- updates `auto_prompt/best_prompt.md` when the selected metric improves

Useful options:

```bash
python auto_prompt/run_iteration.py \
  --iteration 2 \
  --checkpoint /data/feihong/ckpt/model_test_geo_input_ids/last.pth \
  --max-cases 50 \
  --candidate-size 50 \
  --text-score-weight 0.3 \
  --device cuda:0 \
  --overwrite-descriptions
```

For Codex loops, ask it to edit only `auto_prompt/current_prompt.md` using `auto_prompt/prompt_rewrite_guide.md`, then run the iteration script again.
