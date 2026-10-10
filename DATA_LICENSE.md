# Data license and attribution

Files under `data/` and benchmark-derived records under `provenance/live-validation/` and `provenance/hybrid-retrieval/` contain Spider/BIRD-derived questions, SQL, schema catalogs, value catalogs and generated CLARITY annotations. They are distributed under **Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)**, retaining the upstream data license rather than extending the MIT software license to these materials.

License: <https://creativecommons.org/licenses/by-sa/4.0/>. Legal text: <https://creativecommons.org/licenses/by-sa/4.0/legalcode.en>.

Spider's maintainers identify its data license as CC BY-SA 4.0 at <https://yale-lily.github.io/spider>. BIRD's maintainers announced the same data license on April 27, 2024 at <https://bird-bench.github.io/>. These source statements were checked on October 9, 2026. This release does not establish the historical download version or author database snapshot used in the original experiments.

Attribute the CLARITY authors and the upstream benchmark authors, link the license, and identify any changes in redistributions. The source-manifest records this release's changes: final annotations and seeds are byte-identical copies, schema/column/value dictionaries are filtered to the release database IDs, and value dictionaries are compacted and gzip-compressed.

## References

- Tabinda Sarwar, Farhad Moghimifar, Cong Duy Vu Hoang, Xiaoxiao Ma, Shawn Chang Xu, Fahimeh Saleh, Poorya Zaremoodi, Avirup Sil, and Katrin Kirchhoff. 2026. *CLARITY: A Framework and Benchmark for Conversational Language Ambiguity and Unanswerability in Interactive NL2SQL Systems*. <https://arxiv.org/abs/2604.22313>.
- Tao Yu et al. 2018. *Spider: A Large-Scale Human-Labeled Dataset for Complex and Cross-Domain Semantic Parsing and Text-to-SQL Task*. <https://aclanthology.org/D18-1425/>. Dataset: <https://yale-lily.github.io/spider>.
- Jinyang Li et al. 2023. *Can LLM Already Serve as A Database Interface? A BIg Bench for Large-Scale Database Grounded Text-to-SQLs*. <https://arxiv.org/abs/2305.03111>. Dataset: <https://bird-bench.github.io/>.
