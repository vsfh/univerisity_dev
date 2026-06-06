# Auto Research Rules

Use these rules when launching Codex as the outer research loop. Python only evaluates the current prompt once; it does not own the iteration concept.

## Goal

Optimize `auto_prompt/current_prompt.md` so the retrieval score in `auto_prompt/result.csv` improves on the first 50 valid test satellite cases.

Primary metric: `recall@1`.

## Strict Rules

1. Only manually edit `auto_prompt/current_prompt.md` and `auto_prompt/AUTO_RESEARCH_CHANGELOG.md`.
2. Do not modify `tools/generate_qwen_6_4.py`, `dataset.py`, model checkpoints, `/data/feihong/drone_img`, `/data/feihong/img_test_2`, `/data/feihong/ckpt`, or `/data/feihong/hf_cache`.
3. Do not manually edit existing contents of `auto_prompt/result.csv`, `auto_prompt/prompt_history.jsonl`, generated description JSON files, or query record JSONL files. Let the evaluation script append/create them.
4. Do not run `rm`, `mv`, `git reset --hard`, or `git clean`.
5. Before each prompt attempt, search recent top-conference papers or authoritative technical reports about multimodal large models, visual grounding, dense captioning, remote-sensing image-text retrieval, or cross-view geolocalization. Use one theory-backed change direction only.
6. Keep each prompt change focused: prediction format, output constraints, visual evidence hierarchy, spatial relation wording, example organization, or parsing robustness.
7. Preserve the generator output contract: exactly 4 comma-separated English noun phrases, no markdown, no numbering, no headings, no full explanatory sentences.
8. Do not change `max_new_tokens`, image size/pixel parameters, selected 50-case evaluation scope, checkpoint, or model cache settings unless the user explicitly asks.
9. After every prompt edit, update `AUTO_RESEARCH_CHANGELOG.md` with the hypothesis, paper/report inspiration, and expected effect.
10. Run one evaluation:

```bash
python -m auto_prompt.evaluate_prompt
```

11. Read the latest row in `auto_prompt/result.csv` and compare `recall@1` with the previous best row.
12. If `recall@1` improves, keep the prompt and record why it likely helped.
13. If `recall@1` does not improve, record the failure reason and restore `auto_prompt/current_prompt.md` to the previous best prompt or the pre-attempt prompt.
14. If generation/evaluation OOMs or crashes, record the failure reason, restore the pre-attempt prompt, and continue with the next outer-loop attempt.
15. After the requested number of outer-loop attempts, summarize best `recall@1`, kept prompt changes, reverted prompt attempts, and recommended next direction.

## Recommended Codex Launch

```bash
codex exec -C /data/feihong/univerisity_dev -s danger-full-access '
请按照 auto_prompt/auto_research.md 的规则执行 auto-research。

目标：
连续做 20 个外层尝试，优化 auto_prompt/current_prompt.md，使 auto_prompt/result.csv 里的 recall@1 提升。

开始前先读取：
- auto_prompt/auto_research.md
- auto_prompt/current_prompt.md
- auto_prompt/evaluate_prompt.py
- auto_prompt/AUTO_RESEARCH_CHANGELOG.md
- auto_prompt/result.csv 如果存在
- auto_prompt/prompt_history.jsonl 如果存在
'
```
