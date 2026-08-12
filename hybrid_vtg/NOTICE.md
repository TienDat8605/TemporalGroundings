# Third-party notice

Hybrid VTG contains or interoperates with the following research software and models.

## UniVTG

The files under `src/hybrid_vtg/models/univtg/vendor/` are trimmed from the official UniVTG implementation by Kevin Lin and collaborators. The local inference network preserves the official parameter names so released checkpoints can be loaded, and adds sparse absolute-time positional input without adding checkpoint parameters. UniVTG is distributed under the MIT License; see `LICENSES/UniVTG-MIT.txt` and <https://github.com/showlab/UniVTG>.

## TimeLens2

The `coarse-to-fine-64` method is a clean reimplementation of the `embedding-window-local` evaluation design in TimeLens2. No TimeLens2 source tree is required at runtime. The official `MCG-NJU/TimeLens2-4B` checkpoint and current TimeLens2 repository declare Apache License 2.0. The older Tencent TimeLens license retained in `LICENSES/TimeLens-v1.txt` applies to the historical TimeLens project, not the downloaded TimeLens2-4B checkpoint. See <https://github.com/MCG-NJU/TimeLens2> and <https://huggingface.co/MCG-NJU/TimeLens2-4B>.

## TimeLens-7B

`src/hybrid_vtg/models/timelens.py` interoperates with the released
`TencentARC/TimeLens-7B` checkpoint through standard Transformers and
qwen-vl-utils APIs. The native path follows the public two-FPS,
timestamp-interleaved model-card recipe; the evidence path is an independent
integration with this project's adaptive routing and pruning. No upstream
TimeLens source is copied. The checkpoint model card currently labels the
checkpoint BSD-3-Clause, while the official source repository includes the
additional TimeLens terms preserved in `LICENSES/TimeLens-v1.txt`; users must
review both upstream distributions. See <https://github.com/TencentARC/TimeLens> and
<https://huggingface.co/TencentARC/TimeLens-7B>.

## SemVID

`src/hybrid_vtg/models/pruning.py` contains a modified, standalone adaptation of the semantic-oriented Qwen3-VL token selector from the official SemVID project, revision `432a76928817cdfba7d04c460ac475482cd7c3a4`. The adaptation retains SemVID's context/object/motion roles, query-conditioned frame allocation, multi-token object coverage, MMR diversity, and motion scoring, while adding variable per-timestamp capacities and integration with Hybrid VTG's evidence contract. It does not require the SemVID checkout at runtime. SemVID is Copyright 2026 Open Visual-Pruning Suite Authors and is distributed under Apache License 2.0; see `LICENSES/SemVID-Apache-2.0.txt` and <https://github.com/JiaqiLi404/SemVID>.

The compact Qwen prefill adapter also follows SemVID's model-integration approach by preserving caller-provided position IDs during the first generation step after selected visual embeddings are inserted.

## Mage-VL

The encoder-stage policy is a clean-room, training-free adaptation of Mage-VL's dense-anchor/sparse-update principle. It does not include Mage-VL source code, weights, codec tokenizer, or Mage-ViT architecture. It uses decoded-frame optical flow and motion-compensated residuals as a replaceable importance provider, keeps complete Qwen merger cells, and preserves Qwen's original rotary coordinates. See <https://arxiv.org/abs/2607.24904>.

## UniTime

`src/hybrid_vtg/models/unitime.py`, `unitime-fixed`, and `unitime-adaptive` are clean-room implementations of the timestamp-interleaved and coarse-to-fine inference design described by UniTime. They use standard Transformers and PEFT APIs and do not copy or require the UniTime source tree. `unitime-fixed` preserves frozen fixed-segment coarse retrieval; `unitime-adaptive` replaces that retrieval with HMVE top-k corridors. Both may optionally use the independent Mage and SemVID policies. The public `zeqianli/UniTime` adapter is downloaded from Hugging Face at runtime and retains its upstream terms. The UniTime GitHub repository did not expose a license file when this integration was implemented, so no source was copied. See <https://github.com/Lzq5/UniTime>, <https://huggingface.co/zeqianli/UniTime>, and <https://arxiv.org/abs/2506.18883>.

## Models and data

The optional downloader retrieves, but does not redistribute, upstream assets. Qwen, CLIP, PyTorchVideo pretrained weights, UniVTG checkpoints, UniTime, TimeLens, TimeLens2, OMTG, TACoS, and QVHighlights retain their independent licenses and dataset/source-video terms. TACoS is retrieved from VideoMind's BSD-3-Clause dataset repository using its 3 FPS, 480p, no-audio variant; the underlying TACoS video data remains limited to scientific use by MPII and may not be republished. QVHighlights test annotations come from Moment-DETR, and their referenced videos are retrieved as the test-only `jwnt4/qvhighlights-test` archive; the underlying QVHighlights terms remain CC BY-NC-SA 4.0.
