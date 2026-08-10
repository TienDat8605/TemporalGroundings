# Third-party notice

Hybrid VTG contains or interoperates with the following research software and models.

## UniVTG

The files under `src/hybrid_vtg/models/univtg/vendor/` are trimmed from the official UniVTG implementation by Kevin Lin and collaborators. The local inference network preserves the official parameter names so released checkpoints can be loaded, and adds sparse absolute-time positional input without adding checkpoint parameters. UniVTG is distributed under the MIT License; see `LICENSES/UniVTG-MIT.txt` and <https://github.com/showlab/UniVTG>.

## TimeLens2

The `coarse-to-fine-64` method is a clean reimplementation of the `embedding-window-local` evaluation design in TimeLens2. No TimeLens2 source tree is required at runtime. The official `MCG-NJU/TimeLens2-4B` checkpoint and current TimeLens2 repository declare Apache License 2.0. The older Tencent TimeLens license retained in `LICENSES/TimeLens-v1.txt` applies to the historical TimeLens project, not the downloaded TimeLens2-4B checkpoint. See <https://github.com/MCG-NJU/TimeLens2> and <https://huggingface.co/MCG-NJU/TimeLens2-4B>.

## SemVID

No SemVID source code, selector, model subclass, prompt template, or runtime dependency is included. The compact Qwen prefill adapter was informed by SemVID's Apache-2.0 model-integration approach: it preserves caller-provided position IDs during the first generation step after selected visual embeddings are inserted. See <https://github.com/JiaqiLi404/SemVID> and its Apache-2.0 license for the upstream project.

## Models and data

The optional downloader retrieves, but does not redistribute, upstream assets. Qwen, CLIP, PyTorchVideo pretrained weights, UniVTG checkpoints, TimeLens2 checkpoints, OMTG, TACoS, and QVHighlights retain their independent licenses and dataset/source-video terms. TACoS is retrieved from VideoMind's BSD-3-Clause dataset repository using its 3 FPS, 480p, no-audio variant; the underlying TACoS video data remains limited to scientific use by MPII and may not be republished. QVHighlights annotations are CC BY-NC-SA 4.0.
