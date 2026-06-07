# Auto Prompt Research

This folder runs prompt-only research for the Qwen description generator and the trained `Encoder_test` retrieval network.

Default single evaluation:

```bash
python -m auto_prompt.evaluate_prompt
```

What it does:

- reads `auto_prompt/current_prompt.md`
- uses `tools/generate_qwen_6_4.py` through its existing public functions, without modifying that file
- generates descriptions for the first 50 test cases into `auto_prompt/generated_descriptions/`
- loads `/media/data1/feihong/ckpt/model_test_geo_input_ids/last.pth`
- evaluates retrieval with `Encoder_test`
- appends metrics and the full prompt to `auto_prompt/result.csv`
- writes full prompt history to `auto_prompt/prompt_history.jsonl`
- updates `auto_prompt/best_prompt.md` when the selected metric improves

Useful options:

```bash
python -m auto_prompt.evaluate_prompt \
  --checkpoint /media/data1/feihong/ckpt/model_test_geo_input_ids/last.pth \
  --max-cases 50 \
  --candidate-size 50 \
  --text-score-weight 0.3 \
  --device cuda:0 \
  --overwrite-descriptions
```

For Codex loops, use `auto_prompt/auto_research.md`. The outer Codex process owns the iteration loop: it edits `auto_prompt/current_prompt.md`, records the rationale in `AUTO_RESEARCH_CHANGELOG.md`, runs `python -m auto_prompt.evaluate_prompt`, reads `result.csv`, and reverts the prompt when the score does not improve.
