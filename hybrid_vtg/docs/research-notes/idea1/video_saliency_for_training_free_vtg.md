# Video saliency as evidence and a gaze-aware VTG diagnostic

> **Status: separate research direction.** Saliency should first be evaluated as an explanatory metric and a weak auxiliary prior. It should not replace query relevance, motion evidence, or boundary protection, and it must not introduce task-specific training.

## Core idea

A video saliency map estimates where human visual attention is concentrated over space and time. This resembles a moving heatmap:

\[
H_t(x,y)\geq 0,
\qquad
\sum_{x,y}H_t(x,y)=1.
\]

It could help answer two questions relevant to spatial pruning:

1. Does our retained token set preserve the regions that humans attend to?
2. Can a frozen saliency prior protect important objects or interactions that query similarity and motion scoring would otherwise remove?

However, three signals must not be conflated:

- **free-viewing saliency:** where viewers naturally look without a task;
- **task/actor gaze:** where a person looks while performing an action;
- **query relevance:** where a viewer should look to answer a particular language query.

Most video-saliency datasets provide the first signal, while EGTEA Gaze+ provides the second. Neither is automatically equivalent to query-conditioned attention.

## Existing datasets

| Dataset | Attention type and scale | Event spans suitable for VTG? | Best use here |
|---|---|---:|---|
| [DHF1K](https://arxiv.org/abs/1801.07424) | Free-viewing gaze; 1,000 diverse videos, more than 600K frames, 17 observers | No language-grounded spans | Saliency-retention evaluation and frozen saliency-model selection |
| [LEDOV](https://arxiv.org/abs/1709.06316) | Free-viewing gaze; 538 videos, 32 observers | No language-grounded spans | Study object/motion saliency and temporal gaze smoothness |
| [DIEM](https://thediemproject.wordpress.com/videos-and%C2%A0data/) | Free-viewing gaze over dynamic video with raw fixation/saccade records | No language-grounded spans | Generalization and temporal-salience analysis |
| Hollywood-2 / UCF-Sports gaze variants | Viewer gaze on action-centric clips | Mostly trimmed action clips, weak for boundary search | Secondary action-saliency diagnostic |
| [EGTEA Gaze+](https://cbs.ic.gatech.edu/fpv/) | Actor gaze at 30 Hz, 28 hours of cooking, frame-level actions and 10K+ instances | **Yes, after language-query construction** | Primary candidate for a gaze-aware VTG diagnostic |
| [DADA-2000](https://arxiv.org/abs/1904.12634) | Driver attention, accident windows and crash-object locations across 2,000 clips | Yes, but only for accident queries | Secondary domain-specific boundary/attention benchmark |
| [DR(eye)VE](https://arxiv.org/abs/1705.03854) | Driver focus of attention over more than 500K registered frames | Not directly | Domain-specific saliency evaluation |

### Main conclusion from the dataset review

Generic saliency datasets are useful for measuring whether pruning retains human-attended pixels, but they cannot simply be called VTG datasets because they lack language queries and annotated query-event intervals.

EGTEA Gaze+ is the strongest direct conversion candidate because each long session contains synchronized gaze and frame-level human-object action intervals such as "cut bell pepper" or "pour condiment into salad." DADA-2000 offers a smaller, domain-specific alternative with accident windows.

## Track A: use saliency as a pruning metric

This should be the first saliency experiment because it does not change model predictions.

### Saliency-mass retention

Given the retained patch set \(R_t\), measure how much human gaze mass survives pruning:

\[
\operatorname{SMR}
=
\frac{
\sum_t\sum_{p\in R_t}H_{t,p}
}{
\sum_t\sum_p H_{t,p}
}.
\]

Report SMR beside the token-retention ratio. A useful selector should retain substantially more gaze mass than a uniform or random selector using the same number of patches.

Report two versions so temporal and spatial effects are not conflated:

- **Spatial SMR:** compute numerator and denominator only over frames delivered to SemVID. This isolates spatial-token selection.
- **End-to-end SMR:** compute over the complete original video, treating temporally discarded frames as retaining zero mass. This evaluates the combined temporal router and spatial selector.

The primary saliency comparison between official SemVID and a modified spatial selector must use Spatial SMR on exactly the same frames.

### Boundary saliency retention

Measure SMR separately in:

```text
pre-event | start band | event interior | end band | post-event
```

This determines whether the spatial selector preserves human-attended evidence at boundaries or only during the event center.

### Saliency efficiency

\[
\operatorname{SaliencyEfficiency}
=
\frac{\operatorname{SMR}}{\text{retained token fraction}}.
\]

Use this only as a diagnostic, not as a replacement for mIoU. A token set can retain gaze while omitting query-specific evidence.

### Role-level analysis

For SemVID and CAM-SemVID, report the gaze mass captured by:

- object tokens;
- motion tokens;
- context tokens;
- future boundary-anchor tokens.

This can reveal whether the token roles correspond to actual human attention and whether saliency mostly duplicates existing query/motion scores.

## Track B: use frozen saliency as an auxiliary score

Only attempt this after Track A shows that gaze retention correlates with grounding quality or boundary preservation.

Use a publicly pretrained and completely frozen video-saliency model. Do not train or fine-tune it on VTG data. Align its heatmaps with the SemVID patch grid and add it as a bounded auxiliary term:

\[
s_{t,p}
=
s^{\mathrm{SemVID}}_{t,p}
+
\lambda_h g_{t,p}H_{t,p},
\]

where \(g_{t,p}\) is a semantic or boundary gate. The gate is important because free-viewing saliency often favors faces, screen center, high contrast, and unrelated moving objects.

Recommended uses, from safest to most speculative:

1. choose among otherwise similar context-token candidates;
2. break ties between motion tokens with comparable query relevance;
3. protect a small saliency quota in uncertain boundary bands;
4. allocate a modest number of extra tokens to frames where saliency and query evidence agree.

Do not use saliency as the primary object selector. A small static query object may be semantically decisive without attracting generic human gaze.

## Track C: construct EGTEA Gaze-VTG

### Proposed sample construction

Use the untrimmed EGTEA sessions, not only the released trimmed clips. For every annotated action instance:

```json
{
  "video": "session.mp4",
  "query": "When does the person cut the bell pepper?",
  "start": 123.4,
  "end": 127.8,
  "action_label": "cut bell pepper",
  "gaze": "synchronized gaze stream",
  "subject": "subject id",
  "recipe": "recipe id"
}
```

Generate the initial query deterministically from verb/noun labels. Create a small, human-reviewed paraphrase set only if needed; do not use test-video content or model predictions to construct benchmark answers.

### Spatial evidence representation

Synchronize 30 Hz gaze with the 24 Hz video timestamps. Convert valid fixation points into a Gaussian density over the SemVID patch grid. Keep:

- raw gaze point and validity;
- fixation heatmap;
- hand-mask overlap where available;
- distance from gaze to retained object/motion/context patches.

Do not silently replace missing or invalid gaze. Record coverage and exclude only according to a declared rule.

### Boundary-oriented gaze analysis

Actor gaze often anticipates a hand-object interaction. This makes EGTEA useful for testing whether gaze supplies evidence before visual action onset.

Measure:

- gaze shift time relative to annotated action start;
- gaze departure relative to action end;
- gaze retention in pre/start/end/post bands;
- whether gaze helps most for small objects and subtle hand actions;
- whether repeated actions in the same session produce saliency hard negatives.

### Dataset split and leakage rules

- Preserve the official EGTEA train/test splits for comparability.
- Also report a subject-disjoint split if the official protocol permits it.
- Never tune CAM, saliency, or boundary thresholds on the test split.
- A frozen saliency model must not have been trained on test gaze if its predictions are evaluated as a model input.
- Ground-truth gaze is an evaluation annotation, not an inference input.

### Intended status

EGTEA Gaze-VTG should initially be presented as a **diagnostic benchmark**, not a replacement for Charades-STA or ActivityNet-Grounding. It is egocentric, cooking-specific, and records the actor's task gaze rather than an external viewer's query-conditioned gaze.

## Why not convert DHF1K or LEDOV immediately?

They lack natural-language event intervals. Converting them would require new temporal annotations and queries. Automatically generated captions and boundaries could introduce circular evaluation, noisy endpoints, and dependence on the same VLM family being evaluated.

They remain useful for:

- measuring generic human-attention retention;
- selecting a frozen saliency model without VTG labels;
- testing whether the model over-relies on center bias or motion;
- evaluating cross-domain saliency behavior.

A manually annotated DHF1K-VTG subset could be future work, but it should use independent human queries and boundaries rather than pseudo-ground truth.

## Required ablations

At the same temporal components and total token count:

1. official SemVID;
2. SemVID + saliency metric only;
3. SemVID + ungated saliency score;
4. SemVID + query-gated saliency;
5. CAM-SemVID;
6. CAM-SemVID + query-gated saliency;
7. boundary corridor with and without a protected saliency quota.

Report standard VTG metrics together with SMR, boundary SMR, role-level gaze retention, latency, and saliency-model overhead. The important test is whether saliency improves grounding at matched tokens, not whether the retained heatmap looks intuitively appealing.

## Decision

Proceed in this order:

```text
gaze-retention metric on existing selectors
  -> EGTEA Gaze-VTG diagnostic adapter
  -> correlation/error analysis
  -> frozen saliency as a small gated prior
  -> boundary-conditioned saliency quota only if justified
```

This direction is compatible with an absolutely training-free method as long as saliency prediction uses a frozen public checkpoint, ground-truth gaze remains evaluation-only, and no VTG labels are used to fit its weight or thresholds.
