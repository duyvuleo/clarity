# Attribution and release scope

The generation implementation and prompts are provided for the CLARITY research workflow. Author and source provenance are preserved in `CITATION.cff` and `provenance/source-manifest.json`. Copyright notices already present in included source files must be retained. Dependency libraries retain their respective licenses.

Only Spider and BIRD data are included. The source generation methodology, prompt collection and scenario routing are retained subject to the explicit fixes in `CHANGELOG.md`.

Release validation and known limitations are documented in `VERIFICATION.md`.

Optional embedding models are downloaded separately and retain their own licenses. The example uses Apache-2.0-licensed sentence-transformers/all-MiniLM-L6-v2: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2. Model weights are not redistributed in this package.
