# Prompt Rewrite Guide

Use this file as the instruction block when asking Codex to improve `current_prompt.md`.

Goal: improve retrieval Recall@1 on the first 50 test satellite cases without changing `tools/generate_qwen_6_4.py` or the trained checkpoint.

Edit only `auto_prompt/current_prompt.md`. Keep the Qwen output contract stable:

- exactly 4 comma-separated English noun phrases
- no markdown, numbering, headings, or explanatory sentences
- descriptions must be overhead-visible and useful for satellite retrieval
- stay short enough for the dataset tokenizer limit, roughly under 64 text tokens

Good mutation directions:

- Make phrase 1 more discriminative: footprint, roof structure, field/water/road relation, or unusual surrounding pattern.
- Push Qwen away from drone-only cues: facade, viewpoint, sky, lighting, temporary cars, people.
- Encourage cross-view consensus: only mention cues that appear stable across the four directions.
- Prefer layout and geometry over object lists.
- Keep each change small enough that the score can be attributed to the prompt mutation.

After editing, run one evaluation:

```bash
python -m auto_prompt.evaluate_prompt
```

If the new row in `auto_prompt/result.csv` has `accepted=True`, continue mutating from that prompt. Otherwise use `auto_prompt/best_prompt.md` as the next base.
