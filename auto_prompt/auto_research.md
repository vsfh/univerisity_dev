# Auto Research Rules

Use these rules when launching Codex as the outer research loop. Python only evaluates the current prompt once; it does not own the iteration concept.

## Goal

Optimize `auto_prompt/current_prompt.md` so the retrieval score in `auto_prompt/result.csv` improves on the first 50 valid test satellite cases.

Primary metric: `recall@1`.

## Strict Rules

1. Only manually edit `auto_prompt/current_prompt.md` and `auto_prompt/AUTO_RESEARCH_CHANGELOG.md` during prompt attempts.
   Before the first attempt, if the baseline evaluator crashes because of a non-prompt code compatibility bug, make the smallest necessary fix in `auto_prompt/evaluate_prompt.py`, verify it, and record the fix in `AUTO_RESEARCH_CHANGELOG.md`.
2. Do not modify `tools/generate_qwen_6_4.py`, `dataset.py`, model checkpoints, `/media/data1/feihong/drone_img`, `/media/data1/feihong/img_test_2`, `/media/data1/feihong/ckpt`, or `/media/data1/feihong/hf_cache`.
3. Do not manually edit existing contents of `auto_prompt/result.csv`, `auto_prompt/prompt_history.jsonl`, generated description JSON files, or query record JSONL files. Let the evaluation script append/create them.
4. Do not run `rm`, `mv`, `git reset --hard`, or `git clean`.
5. Before each prompt attempt, search recent top-conference papers or authoritative technical reports about multimodal large models, visual grounding, dense captioning, remote-sensing image-text retrieval, or cross-view geolocalization. Use one theory-backed change direction only.
6. Keep each prompt change focused: prediction format, output constraints, visual evidence hierarchy, spatial relation wording, example organization, or parsing robustness.
7. Preserve the generator output contract: exactly 4 comma-separated English noun phrases, no markdown, no numbering, no headings, no full explanatory sentences.
8. Do not change `max_new_tokens`, image size/pixel parameters, selected 50-case evaluation scope, checkpoint, or model cache settings unless the user explicitly asks.
   For every new prompt attempt, generate Qwen descriptions only through `python -m auto_prompt.evaluate_prompt --max-cases 50 --description-dir auto_prompt/generated_descriptions/auto_research_loop --overwrite-descriptions --qwen-generate-batch-size 16 --retrieval-query-source text`; never call `tools/generate_qwen_6_4.py` directly over the full drone image root.
   This may overwrite JSON files inside `auto_prompt/generated_descriptions/auto_research_loop`, but must never overwrite original per-case description JSON files under `/media/data1/feihong/drone_img`.
9. After every prompt edit, update `AUTO_RESEARCH_CHANGELOG.md` with the hypothesis, paper/report inspiration, and expected effect.
10. Run one evaluation:

```bash
python -m auto_prompt.evaluate_prompt --max-cases 50 --description-dir auto_prompt/generated_descriptions/auto_research_loop --overwrite-descriptions --qwen-generate-batch-size 16 --retrieval-query-source text
```

11. Read the latest row in `auto_prompt/result.csv` and compare `recall@1` with the previous best row.
12. If `recall@1` improves, keep the prompt and record why it likely helped.
13. If `recall@1` does not improve, record the failure reason and restore `auto_prompt/current_prompt.md` to the previous best prompt or the pre-attempt prompt.
14. If generation/evaluation OOMs or crashes, record the failure reason, restore the pre-attempt prompt, and continue with the next outer-loop attempt.
15. After the requested number of outer-loop attempts, summarize best `recall@1`, kept prompt changes, reverted prompt attempts, and recommended next direction.

## Recommended Codex Launch

```bash
codex exec -C /media/data1/feihong/univerisity_dev -s danger-full-access '
请按照 auto_prompt/auto_research.md 的规则执行 auto-research。

目标：
连续做 20 个外层尝试，优化 auto_prompt/current_prompt.md，使 auto_prompt/result.csv 里的 recall@1 提升。

执行要求：
- 先运行一次基线评估。如果评估在 prompt 无关的代码兼容问题上崩溃，允许先对 auto_prompt/evaluate_prompt.py 做最小修复并记录到 AUTO_RESEARCH_CHANGELOG.md，然后继续 20 个 prompt 外层尝试。
- 每次尝试只运行 python -m auto_prompt.evaluate_prompt --max-cases 50 --description-dir auto_prompt/generated_descriptions/auto_research_loop --overwrite-descriptions --qwen-generate-batch-size 16 --retrieval-query-source text。
- 每个新 prompt 的 Qwen 描述只能为这 50 个测试 case 生成；不要直接调用 tools/generate_qwen_6_4.py 扫描 /media/data1/feihong/drone_img 全量目录。
- 新描述只能覆盖 auto_prompt/generated_descriptions/auto_research_loop 里的本轮测试缓存，不允许覆盖 /media/data1/feihong/drone_img 下已有的原始描述 JSON。
- 除上述评估脚本兼容修复外，prompt 尝试期间只手动编辑 auto_prompt/current_prompt.md 和 auto_prompt/AUTO_RESEARCH_CHANGELOG.md。

开始前先读取：
- auto_prompt/auto_research.md
- auto_prompt/current_prompt.md
- auto_prompt/evaluate_prompt.py
- auto_prompt/AUTO_RESEARCH_CHANGELOG.md
- auto_prompt/result.csv 如果存在
- auto_prompt/prompt_history.jsonl 如果存在
'
```
