# Third-party notice

Hybrid VTG contains or interoperates with the following research software and models.

## UniVTG

The files under `src/hybrid_vtg/models/univtg/vendor/` are trimmed from the official UniVTG implementation by Kevin Lin and collaborators. The local inference network preserves the official parameter names so released checkpoints can be loaded, and adds sparse absolute-time positional input without adding checkpoint parameters. UniVTG is distributed under the MIT License; see `LICENSES/UniVTG-MIT.txt` and <https://github.com/showlab/UniVTG>.

## TimeLens2

The `coarse-to-fine-64` method is a clean reimplementation of the `embedding-window-local` evaluation design in TimeLens2. No TimeLens2 source tree is required at runtime. The optional `MCG-NJU/TimeLens2-4B` checkpoint remains subject to the TimeLens license, which restricts use to academic purposes and states that TimeLens is not intended for use within the European Union. See `LICENSES/TimeLens2.txt` and <https://github.com/MCG-NJU/TimeLens2>.

## SemVID

No SemVID source code, selector, model subclass, prompt template, or runtime dependency is included. The compact Qwen prefill adapter was informed by SemVID's Apache-2.0 model-integration approach: it preserves caller-provided position IDs during the first generation step after selected visual embeddings are inserted. See <https://github.com/JiaqiLi404/SemVID> and its Apache-2.0 license for the upstream project.

## Models and data

Qwen, CLIP, PyTorchVideo pretrained weights, UniVTG checkpoints, TimeLens2 checkpoints, OMTG, TACoS, and QVHighlights are downloaded separately. Their licenses and dataset/source-video terms apply independently.
