# Auto Research Changelog

Record each Codex-driven prompt attempt here. The evaluation script records full prompts and metrics in `result.csv` and `prompt_history.jsonl`; this changelog is for the research rationale and rollback notes.

## 2026-06-07 Reset before 15-attempt loop

- Removed previous auto-research artifacts from `auto_prompt/result.csv`, `auto_prompt/prompt_history.jsonl`, `auto_prompt/best_prompt.md`, `auto_prompt/query_records/`, `auto_prompt/prompt_archive/`, and `auto_prompt/generated_descriptions/auto_research_loop/`.
- Preserved `auto_prompt/current_prompt.md` as the starting prompt for this new run.
- Target: run a fresh 15-attempt auto-research loop on the first 50 valid test cases, using text-feature retrieval.

## 2026-06-07 Baseline

- Prompt SHA: `a61c45559d`; metric: recall@1 `0.1450`, recall@5 `0.4350`, recall@10 `0.6150`.
- No compatibility fix was needed; the baseline command completed and only generated descriptions under `auto_prompt/generated_descriptions/auto_research_loop/`.

## Attempt 1 - exact separator contract

- Hypothesis: baseline descriptions sometimes merged supporting cues into sentence-like text, so enforcing exactly three commas and no internal phrase commas should make the SigLIP text query more consistently parse as four compact retrieval phrases.
- Paper/report inspiration: recent remote-sensing image-text retrieval reviews emphasize fine-grained semantic alignment and structured high-quality text as a way to reduce the visual-language semantic gap.
- Expected effect: fewer malformed long captions, more stable phrase-level cues for text-only retrieval.
- Result: rejected. recall@1 `0.1000` vs previous best `0.1450`; likely over-constrained formatting reduced descriptive content and useful spatial context. Restored the baseline prompt.

## Attempt 2 - compass-relative landmark wording

- Hypothesis: text-guided cross-view localization benefits from localization details, so asking for nearest map landmark plus explicit north/east/south/west adjacency should improve discrimination among visually similar campus buildings.
- Paper/report inspiration: CrossText2Loc / CVG-Text reports gains from LMM-generated scene descriptions with localization details for satellite/OSM retrieval.
- Expected effect: stronger map-relative cues without reducing phrase richness.
- Result: accepted. recall@1 `0.1600` vs previous best `0.1450`; likely helped by retaining rich captions while emphasizing map-relative landmark adjacency.

## Attempt 3 - geometry-first evidence hierarchy

- Hypothesis: shifting supporting cues from mixed semantic/color evidence toward footprint outline, roof orientation, boundary shape, and topology should better align text with satellite grid features.
- Paper/report inspiration: GeoGround and GeoPriorCLIP emphasize geometry-guided learning, boundaries, and topological/spatial priors for remote-sensing visual grounding and retrieval.
- Expected effect: less confusion between similarly colored campus buildings; more map-verifiable geometry in the text query.
- Result: rejected. recall@1 `0.0950` vs previous best `0.1600`; geometry-first wording over-suppressed color/material and semantic landmarks that the text encoder apparently uses. Restored Attempt 2 prompt.

## Attempt 4 - dual-granularity cue slots

- Hypothesis: assigning the three supporting phrases to local object evidence, scene context, and relational layout should preserve richer dual-granularity semantics while avoiding repetitive cue types.
- Paper/report inspiration: LRSCLIP / DGTRSD argue that remote-sensing retrieval benefits from both short and long descriptions to address semantic granularity limitations.
- Expected effect: better coverage of discriminative local and global evidence in each four-phrase query.
- Result: rejected. recall@1 `0.1500` vs previous best `0.1600`; fixed cue slots were slightly too rigid and likely prevented Qwen from selecting the most salient evidence per scene. Restored Attempt 2 prompt.

## Attempt 5 - contrastive discriminative detail

- Hypothesis: explicitly asking for details that distinguish the site from similar nearby campus buildings should increase the semantic gap among hard negatives without changing the output format.
- Paper/report inspiration: IJCAI 2025 DFIM identifies semantic confusion from redundant visual representations and high inter-class similarity as a key RS image-text retrieval failure mode.
- Expected effect: fewer generic campus/building phrases and stronger hard-negative separation.
- Result: rejected. recall@1 `0.1150` vs previous best `0.1600`; contrastive wording likely pushed Qwen toward abstract hard-negative reasoning instead of direct map-visible evidence. Restored Attempt 2 prompt.

## Attempt 6 - multi-scale target and context

- Hypothesis: asking for both close target details and wider surrounding layout should capture complementary local/global semantics without rigidly assigning phrase slots.
- Paper/report inspiration: MSSA highlights multi-scale semantic-aware alignment and the need to represent both local details and global structures in remote-sensing image-text retrieval.
- Expected effect: richer descriptions that preserve target-specific cues while adding map-scale context.
- Result: rejected. recall@1 `0.1100` vs previous best `0.1600`; wider layout improved rank-list breadth but hurt top1 specificity. Restored Attempt 2 prompt.

## Attempt 7 - countable repeated structures

- Hypothesis: encouraging countable repeated structures such as twin roofs, multiple vents, rows, and courts should strengthen fine-grained minimal-change cues while preserving direct overhead evidence.
- Paper/report inspiration: NeurIPS 2024 VisMin and 2024 RS captioning work on object counts both identify object count, attributes, and spatial relation as important fine-grained visual-language distinctions.
- Expected effect: better separation between visually similar roofs and building groups with repeated elements.
- Result: rejected. recall@1 `0.0950` vs previous best `0.1600`; count emphasis likely distracted from the primary overhead anchor. Restored Attempt 2 prompt.

## Attempt 8 - anchor attribute plus spatial relation

- Hypothesis: making the first phrase carry shape, roof material/color, and nearest landmark should combine object attributes with spatial relation in the highest-weight anchor phrase.
- Paper/report inspiration: WACV 2025 Learning Visual Grounding from Generative VLM explicitly models important attributes and inter-object spatial relations in generated referring expressions.
- Expected effect: more discriminative first phrase without changing supporting cue freedom.
- Result: pending.
