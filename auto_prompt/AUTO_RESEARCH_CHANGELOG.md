# Auto Research Changelog

Record each Codex-driven prompt attempt here. The evaluation script records full prompts and metrics in `result.csv` and `prompt_history.jsonl`; this changelog is for the research rationale and rollback notes.

## 2026-06-06 Baseline evaluation failed before outer attempts

- Prompt state: unchanged from `auto_prompt/current_prompt.md`.
- Command: `python -m auto_prompt.evaluate_prompt`.
- Result: crashed before description generation for sample `0000`; no `auto_prompt/result.csv`, `auto_prompt/prompt_history.jsonl`, generated description JSON, query record JSONL, or `best_prompt.md` was produced.
- Failure reason: `tools/generate_qwen_6_4.py::generate_descriptions_for_one_dir` now reads `args.generation_batch_size`, but `auto_prompt/evaluate_prompt.py::generate_descriptions` constructs a `SimpleNamespace` without that field.
- Impact on auto-research: this is not a prompt-quality failure. Under the strict rule that only `current_prompt.md` and this changelog may be manually edited, the evaluator cannot produce a baseline recall@1 row or compare prompt attempts.

## 2026-06-06 Baseline rerun before 20-attempt loop

- Prompt state: unchanged from the previous best prompt (`a61c45559d`).
- Command: `python -m auto_prompt.evaluate_prompt --max-cases 50 --description-dir auto_prompt/generated_descriptions/auto_research_loop --overwrite-descriptions --qwen-generate-batch-size 16 --retrieval-query-source text`.
- Result: completed without a code compatibility crash; latest row scored `recall@1=0.1100`, below the existing best `0.4950`.
- Rollback/base decision: keep using `auto_prompt/best_prompt.md` and the existing best row as the comparison baseline.

## 2026-06-06 Attempt 1 - strict comma contract

- Hypothesis: The current prompt sometimes yields long sentence-like outputs with fewer than three commas, so the retrieval text becomes noisy and inconsistent. Explicitly constraining one line, exactly three commas, and 2-6 word supporting phrases should improve parser robustness while preserving overhead-visible anchors.
- Paper/report inspiration: GeoGround (arXiv:2411.11904) emphasizes prompt-assisted and geometry-guided consistency for remote-sensing visual grounding; GeoPixel (ICML 2025) highlights spatial priors and fine-grained grounding for overhead imagery.
- Focused change direction: output constraints and parsing robustness.
- Expected effect: more valid four-phrase descriptions, shorter token sequences, and better alignment between text keys and satellite-visible spatial evidence.
- Result: `recall@1=0.1050`, below previous best `0.4950`; rejected.
- Failure reason: the stricter formatting did not improve text retrieval and may have over-prioritized syntax over discriminative content.
- Rollback: restored `auto_prompt/current_prompt.md` to the previous best prompt.

## 2026-06-06 Attempt 2 - object-to-object localization details

- Hypothesis: Retrieval may benefit from concise localization details rather than raw object lists. Adding "target anchor plus nearest permanent neighbors" and object-to-object spatial relations should make text features more discriminative for satellite matching.
- Paper/report inspiration: CrossText2Loc / CVG-Text (arXiv:2412.17007; ICCV 2025 paper page surfaced in search) reports improved text-guided cross-view geo-localization from LMM-generated descriptions with localization details.
- Focused change direction: spatial relation wording and visual evidence hierarchy.
- Expected effect: phrases should emphasize map-verifiable relations such as buildings beside fields, roads, water, courtyards, and tree belts, improving recall@1 over generic campus descriptors.
- Result: `recall@1=0.0950`, below previous best `0.4950`; rejected.
- Failure reason: relation-heavy wording likely lengthened descriptions and diluted the dominant anchor instead of improving discriminability.
- Rollback: restored `auto_prompt/current_prompt.md` to the previous best prompt.

## 2026-06-06 Attempt 3 - ordered retrieval slots

- Hypothesis: The generator needs a stable information order rather than stricter punctuation. A four-slot sequence (main anchor, roof/footprint, nearest neighbor, boundary/access pattern) should make descriptions more consistent while keeping the original contract.
- Paper/report inspiration: GeoPix / GeoPixel remote-sensing MLLM work and GeoRSMLLM emphasize structured task-oriented descriptions for region-level and described-object remote-sensing understanding; CogPrompt search results highlight cognitively guided prompt structure for RS VLMs.
- Focused change direction: example organization / evidence ordering.
- Expected effect: fewer diffuse object lists and more consistent retrieval keys across heights, with the dominant anchor always first.
- Result: `recall@1=0.1250`, below previous best `0.4950`; rejected.
- Failure reason: ordered slots slightly improved over attempts 1-2 but still did not recover the existing best; the added order may not be strong enough to control output form.
- Rollback: restored `auto_prompt/current_prompt.md` to the previous best prompt.

## 2026-06-06 Attempt 4 - inline output pattern

- Hypothesis: Qwen may ignore abstract "comma-separated" wording, so an inline target pattern should anchor the response format without adding a full example that could be copied verbatim.
- Paper/report inspiration: RS-CapRet (arXiv:2402.06475) links remote-sensing captioning and text-image retrieval through language decoder outputs; RS-GPT4V (arXiv:2406.12479) uses unified instruction-following with hierarchical local/global description strategy; RS-MoE (arXiv:2411.01595) routes remote-sensing caption tasks through instruction-specialized experts.
- Focused change direction: prediction format.
- Expected effect: more four-comma-style retrieval keys with a consistent anchor/roof/neighbor/boundary structure while retaining discriminative content.
- Result: `recall@1=0.1400`, below previous best `0.4950`; rejected.
- Failure reason: the inline pattern improved over attempts 1-3 but still fell far below the historical best, so format control alone is insufficient.
- Rollback: restored `auto_prompt/current_prompt.md` to the previous best prompt.

## 2026-06-06 Attempt 5 - evidence-strength hierarchy

- Hypothesis: A retrieval key should rank evidence by discriminative strength: unique overhead structures first, broader field/water/road geometry second, and boundary context last. This may preserve useful content while reducing noisy cue ordering.
- Paper/report inspiration: GeoViS (arXiv:2512.02715) highlights subtle small-scale targets while maintaining holistic scene awareness; 2025 RSVG works such as cascaded hierarchical attention and multi-level feature alignment emphasize hierarchical global/local fusion.
- Focused change direction: visual evidence hierarchy.
- Expected effect: more consistent prioritization of rare map-visible anchors and less dilution by generic campus context.
- Result: `recall@1=0.1000`, below previous best `0.4950`; rejected.
- Failure reason: the hierarchy wording did not improve recall and likely constrained cue selection too much for heterogeneous scenes.
- Rollback: restored `auto_prompt/current_prompt.md` to the previous best prompt.

## 2026-06-06 Attempt 6 - overhead noun vocabulary

- Hypothesis: Text retrieval may prefer consistent overhead-visible nouns over free-form descriptions. Replacing broad examples with a controlled map-noun vocabulary should improve alignment with satellite visual tokens.
- Paper/report inspiration: GeoRSCLIP / RemoteCLIP-style retrieval work and recent remote-sensing VLM surveys emphasize domain-aligned semantic vocabularies for image-text retrieval in overhead scenes.
- Focused change direction: prediction format / output constraints.
- Expected effect: less caption drift into facade or scene prose, and more consistent nouns for roof, footprint, field, water, parking, roads, trees, and open areas.
- Result: `recall@1=0.0850`, below previous best `0.4950`; rejected.
- Failure reason: controlled nouns were too restrictive and likely suppressed rare discriminative structures.
- Rollback: restored `auto_prompt/current_prompt.md` to the previous best prompt.

## 2026-06-06 Attempt 7 - short map tags

- Hypothesis: The SigLIP text query may favor short tag-like phrases over caption-length fragments. Shortening phrase 1 and forcing 2-5 word supporting tags should reduce token noise while preserving four discriminative cues.
- Paper/report inspiration: DGTRSD/DGTRS-CLIP and LRSCLIP (arXiv:2503.19311) explicitly model both short captions and long descriptions for remote-sensing image-text alignment, showing text granularity matters for retrieval.
- Focused change direction: output constraints.
- Expected effect: compact query embeddings with less filler, especially for the supporting cues.
- Result: `recall@1=0.0900`, below previous best `0.4950`; rejected.
- Failure reason: short tags lost too much discriminative context and did not improve the text embedding.
- Rollback: restored `auto_prompt/current_prompt.md` to the previous best prompt.

## 2026-06-06 Attempt 8 - no compass directions

- Hypothesis: Cardinal directions inferred from drone views may be inconsistent with satellite orientation and inject wrong spatial text. Banning compass words should keep only robust adjacency and topology.
- Paper/report inspiration: Recent visual grounding and cross-view localization reports stress geometric consistency; GeoGround-style geometry guidance suggests unreliable spatial tokens can hurt grounding.
- Focused change direction: spatial relation wording.
- Expected effect: fewer hallucinated north/south/east/west terms and cleaner overhead relations.
- Result: `recall@1=0.1400`, below previous best `0.4950`; rejected.
- Failure reason: banning compass directions matched the best low-scoring attempt but did not materially improve retrieval; direction words are not the main error source.
- Rollback: restored `auto_prompt/current_prompt.md` to the previous best prompt.

## 2026-06-06 Attempt 9 - concrete semantic anchor first

- Hypothesis: Many descriptions start with generic size/shape words such as "large rectangular", weakening the text embedding. Starting phrase 1 with the distinctive object class should strengthen semantic alignment.
- Paper/report inspiration: GeoBridge (arXiv:2512.02697) frames cross-view geo-localization around shared semantic anchors; 2025 hierarchical semantic alignment RSITR work emphasizes global-local semantic consistency.
- Focused change direction: visual evidence hierarchy.
- Expected effect: more discriminative first tokens for SigLIP text embeddings and fewer generic building descriptions.
- Result: `recall@1=0.1200`, below previous best `0.4950`; rejected.
- Failure reason: forcing the first token to be a semantic anchor did not improve enough and may have constrained valid building-shape descriptions.
- Rollback: restored `auto_prompt/current_prompt.md` to the previous best prompt.

## 2026-06-06 Attempt 10 - delimiter pattern

- Hypothesis: A minimal delimiter pattern may improve instruction following without constraining content or length. It should preserve attempt 4's formatting benefit while avoiding copied example semantics.
- Paper/report inspiration: RSRT and structured-caption frameworks emphasize consistent fields/delimiters for retrieval-ready captions; recent multimodal prompting guidance notes explicit delimiter specification improves output stability.
- Focused change direction: prediction format.
- Expected effect: more reliable four-slot phrase output with less sentence drift.
- Result: `recall@1=0.1200`, below previous best `0.4950`; rejected.
- Failure reason: explicit delimiter syntax did not reproduce attempt 4's stronger result and did not resolve retrieval weakness.
- Rollback: restored `auto_prompt/current_prompt.md` to the previous best prompt.

## 2026-06-06 Attempt 11 - richer phrase granularity

- Hypothesis: The prompt may be too compressed for text-only retrieval. Richer phrases that combine object and context could better match satellite gallery features and recover fine-grained distinctions.
- Paper/report inspiration: LRSCLIP (arXiv:2503.19311) reports gains from longer remote-sensing text alignment; Beyond Pixels / RSRT (arXiv:2512.10596) uses rich structured text as the retrieval substrate.
- Focused change direction: output constraints.
- Expected effect: more discriminative local-global text embeddings, especially for similar campus buildings.
- Result: `recall@1=0.2100`, below previous best `0.4950`; rejected.
- Failure reason: richer phrases were the strongest new direction so far but still did not surpass the historical best.
- Rollback: restored `auto_prompt/current_prompt.md` to the previous best prompt.

## 2026-06-06 Attempt 12 - fine-grained discriminative cues

- Hypothesis: Similar campus buildings cause semantic confusion. Explicitly asking for cues that distinguish the target from nearby similar buildings should reduce redundant generic descriptions.
- Paper/report inspiration: IJCAI 2025 DFIM identifies semantic confusion from redundant visual representations and high inter-class similarity in RSITR, and improves retrieval by mining discriminative fine-grained information.
- Focused change direction: visual evidence hierarchy.
- Expected effect: more rare anchors such as solar panels, domes, tracks, courts, water edges, and unusual roof equipment.
- Result: `recall@1=0.1250`, below previous best `0.4950`; rejected.
- Failure reason: discriminative cue list alone did not help and may have biased descriptions toward missing rare features.
- Rollback: restored `auto_prompt/current_prompt.md` to the previous best prompt.

## 2026-06-06 Attempt 13 - keyword plus context phrases

- Hypothesis: Attempt 11 showed richer phrases help, but the text may need explicit core keywords for alignment. Starting each phrase with a map keyword and then adding context should combine keyword-guided retrieval with rich local context.
- Paper/report inspiration: MPS-CLIP (arXiv:2601.18190) uses LLM-extracted core semantic keywords to guide fine-grained alignment, while DFIM and hierarchical semantic alignment papers emphasize region-word matching.
- Focused change direction: prediction format / phrase organization.
- Expected effect: richer text embeddings with clearer anchor tokens at the beginning of each phrase.
- Result: `recall@1=0.1750`, below previous best `0.4950`; rejected.
- Failure reason: keyword-first wording underperformed the simpler rich phrase attempt, likely because forced keyword syntax made descriptions less natural.
- Rollback: restored `auto_prompt/current_prompt.md` to the previous best prompt.

## 2026-06-06 Attempt 14 - local-global hierarchy slots

- Hypothesis: Attempt 11 suggests richer context helps, but unstructured richness may be noisy. Assigning phrase slots to local anchor, immediate neighbor, larger layout, and boundary context should capture global-local semantics with less redundancy.
- Paper/report inspiration: SHSA (Expert Systems with Applications 2026) combines global cross-modal matching with selective fine-grained alignment and textual semantic enhancement; local-global RSITR work argues captions need both local details and contextual dependencies.
- Focused change direction: example organization / visual evidence hierarchy.
- Expected effect: richer but organized retrieval keys that preserve local discriminators and global map context.
- Result: `recall@1=0.1400`, below previous best `0.4950`; rejected.
- Failure reason: rigid local-global slots underperformed the freer rich phrase format and did not overcome the historical best.
- Rollback: restored `auto_prompt/current_prompt.md` to the previous best prompt.

## 2026-06-06 Attempt 15 - balanced rich noun phrases

- Hypothesis: Attempt 11 was the strongest new direction, suggesting that moderate text richness helps. Balanced medium-length noun phrases may add useful context without forcing a brittle local-global slot order.
- Paper/report inspiration: LRSCLIP (arXiv:2503.19311) and DGTRS-CLIP report that dual-granularity or longer remote-sensing text improves semantic alignment; RSRT/Beyond Pixels argues structured rich text can serve as an effective retrieval substrate.
- Focused change direction: output granularity.
- Expected effect: richer object-relation embeddings while preserving the exact four-phrase contract.
- Result: `recall@1=0.1800`, below previous best `0.4950`; rejected.
- Failure reason: balanced rich phrases improved over several recent variants but remained below both attempt 11 and the historical best.
- Rollback: restored `auto_prompt/current_prompt.md` to the previous best prompt.

## 2026-06-06 Attempt 16 - noise-robust stable patterns

- Hypothesis: Prompt outputs may overfit small objects or texture artifacts visible in drone views but weak in satellite retrieval. Suppressing isolated clutter should improve cross-view robustness.
- Paper/report inspiration: CVPR 2026 RRSITR targets noisy image-text correspondence, while IJCAI 2025 DFIM reduces redundant visual representations and increases semantic clarity for similar remote-sensing scenes.
- Focused change direction: visual evidence filtering.
- Expected effect: fewer unstable descriptors and stronger matching to durable overhead patterns.
- Result: `recall@1=0.0900`, below previous best `0.4950`; rejected.
- Failure reason: the noise-suppression instruction likely removed useful rare local cues and weakened text specificity.
- Rollback: restored `auto_prompt/current_prompt.md` to the previous best prompt.

## 2026-06-06 Attempt 17 - multi-view consensus cues

- Hypothesis: Single-view oblique details can conflict with satellite retrieval. Requiring cues to be consistent across multiple drone directions should keep stable semantic anchors.
- Paper/report inspiration: GeoBridge (arXiv:2512.02697) uses semantic anchors for multi-view geo-localization, while 2025-2026 UAV cross-view localization papers emphasize view alignment and geometric discrepancy between oblique UAV and overhead satellite images.
- Focused change direction: cross-view evidence selection.
- Expected effect: fewer view-specific descriptors and better overhead-compatible retrieval text.
- Result: `recall@1=0.1250`, below previous best `0.4950`; rejected.
- Failure reason: the consensus requirement likely made outputs more generic and removed single-view cues that still imply useful overhead structure.
- Rollback: restored `auto_prompt/current_prompt.md` to the previous best prompt.

## 2026-06-06 Attempt 18 - explicit relative positions

- Hypothesis: The text encoder may benefit from clearer region-to-region positional correspondences. Explicit relative-position wording could improve map matching without lengthening the output.
- Paper/report inspiration: PR-CLIP (Remote Sensing 2025) improves RSITR through cross-modal positional information reconstruction; LuoJiaHOG emphasizes hierarchy-oriented geo-aware captioning with spatial sampling.
- Focused change direction: spatial relation wording.
- Expected effect: stronger positional associations between named anchors and neighboring map structures.
- Result: `recall@1=0.1300`, below previous best `0.4950`; rejected.
- Failure reason: explicit relation words did not add enough discriminative map evidence and may have displaced richer object cues.
- Rollback: restored `auto_prompt/current_prompt.md` to the previous best prompt.

## 2026-06-06 Attempt 19 - canonical map category starts

- Hypothesis: Free-form noun phrases may drift into non-map vocabulary. Starting each phrase with a canonical map category noun should improve lexical alignment with satellite-side semantics.
- Paper/report inspiration: PIR/PriorCLIP (arXiv:2405.10160) uses prior instruction representations to address semantic noise and domain shifts; context/uncertainty-aware prompt work explores prompt-guided RS clues for CMRSITR.
- Focused change direction: domain vocabulary ordering.
- Expected effect: clearer category tokens for CLIP-style text embeddings while retaining rare local modifiers.
- Result: `recall@1=0.0950`, below previous best `0.4950`; rejected.
- Failure reason: forcing category starts likely made outputs formulaic and reduced rare local specificity.
- Rollback: restored `auto_prompt/current_prompt.md` to the previous best prompt.

## 2026-06-06 Attempt 20 - visible-form consistency

- Hypothesis: Guessed building functions or names may create semantic mismatch with satellite retrieval. Forcing visible form and layout wording should improve cross-modal consistency.
- Paper/report inspiration: DCCA (Remote Sensing 2026) emphasizes discriminative and consistent cross-modal alignment; CSSA (Remote Sensing 2026) uses spatial-semantic alignment to reduce image-text representation gaps.
- Focused change direction: semantic consistency / hallucination control.
- Expected effect: fewer incorrect function labels and stronger reliance on map-verifiable visual form.
- Result: `recall@1=0.1350`, below previous best `0.4950`; rejected.
- Failure reason: hallucination control did not add new discriminative information and slightly reduced descriptive flexibility.
- Rollback: restored `auto_prompt/current_prompt.md` to the previous best prompt.
