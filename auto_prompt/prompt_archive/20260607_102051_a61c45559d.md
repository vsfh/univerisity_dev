You are writing a compact retrieval key for matching drone-view images to a satellite map.

The four input images show the same target area at one flight height from north, east, south, and west views. First compare the four views, then keep only evidence that should remain visible from overhead.

Output exactly 4 comma-separated English noun phrases and nothing else.

Phrase 1 must be 10-14 words: name the most discriminative overhead-visible anchor, including its footprint, roof, or surrounding landmark relation.
Phrases 2-4 must be short supporting cues: use stable layout evidence such as building shape, roof color/texture, courtyard/open space, sports field, water edge, parking pattern, road geometry, tree belt, or relative position.

Prefer rare, local, map-verifiable details over generic campus words. Avoid facade, camera angle, lighting, sky, people, vehicles as temporary objects, uncertainty words, markdown, numbering, headings, and full sentences.
