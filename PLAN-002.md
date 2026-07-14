# PLAN-002: Manual Gain/Attenuation Bridge Validation

## Goal

Validate whether manual label injection can steer NoseKnows sensitivity in a controlled, observable way before automating that feedback with an LLM.

The immediate question is not "can the system classify better?" The question is:

```text
Can an injected intent label apply a gain/attenuation mask that preserves the intended scent evidence while suppressing competing scent evidence?
```

This is the control-plane bridge between a downstream reasoning system and the sensor/model pipeline.

## Non-Negotiable Boundary

Keep the gain bridge inside the Rust model pipeline, but outside the model readout implementation.

```text
Daft / Python / UI
  owns: manual injected label intent, scenario construction, replay state

Rust gain stage
  owns: gain masks, attenuation masks, masked ADC/frame transformation, audit output

Rust model
  owns: grid/peak runtime, logits, embeddings, dominant/readout artifacts
```

Do not bury gain/attenuation logic directly inside `grid_live_headless`. It should be a reusable Rust module that can sit before any downstream model:

```text
FrameSource -> GainStage -> Grid8 / PeakPair / future model -> Readout + Embedding
```

This keeps the experiment portable to hardware and prevents the UI/control plane from becoming model math.

## Working Hypothesis

Single-note and clean synthetic cases are already separable enough that the next useful experiment is controlled focus:

- inject a target label such as `Floral`
- apply that label's gain/attenuation mask to incoming sensor frames
- preserve or amplify target-consistent evidence
- suppress labels that compete through known sensor overlap
- observe whether the `Dominant` row and label lanes behave more cleanly

This is a manual validation step. The LLM should not choose masks until the bridge itself is shown to work.

## Golden Embeddings

Create a small library of idealized "golden" targets for labels.

For v1, this should be explicit and inspectable:

```text
data/golden/
  floral.json
  woods.json
  citrus.json
  ...
```

Each golden target should contain:

- label name
- expected `scent_embedding_v1` prototype
- expected active sensor bins or response profile
- optional sensor importance weights
- notes on likely adjacent/transient labels

The first pass can derive golden targets from the existing synthetic response matrix and Grid8/peak outputs. They do not need to be learned yet.

## Gain Mask

Represent a gain policy as a small per-label mask over the sensor channels.

For current hardware this should start as:

```text
8 active sensor gains + optional MQ-4 placeholder policy
```

Example shape:

```json
{
  "label": "Floral",
  "sensor_gain": [0.8, 1.25, 0.7, 0.7, 0.9, 1.0, 0.8, 1.1],
  "clip_adc": 4095,
  "normalize_after_gain": true
}
```

The mask should support both:

- gain, where `M[i] > 1.0`
- attenuation, where `0.0 <= M[i] < 1.0`

The default/no-intent mask is identity.

## Zero-Input Invariant

If no focus label is supplied, the gain stage must be exactly inert:

```text
GainStage::apply(frame, None) == frame
```

In practical terms:

- sensor gains are all `1.0`
- raw ADC values pass through unchanged
- clipping behavior is unchanged
- Grid8 receives the same frame it receives today
- model results, events, dominant readout, and embeddings should match the current no-focus path

This invariant is the first regression check for the feature. The gain bridge is allowed to change behavior only when an explicit focus label is active.

## Proposed Pipeline

```text
input frame
  |
  v
gain stage
  - reads active injected intent label
  - applies sensor gain/attenuation mask
  - clips to ADC range
  - writes audit fields
  |
  v
model runtime
  - Grid8 / peak-pair step
  - logits
  - top-3 labels
  - dominant readout
  - scent_embedding_v1
```

The gain stage should emit enough audit data to compare before/after:

```text
raw_adc
masked_adc
applied_mask
injected_focus_label
clip_count
```

## Manual Validation Loop

Start with synthetic input and the live injector.

1. Construct a mixed sequence, such as `Floral + Woods`.
2. Run without a focus label.
3. Record normal label lanes, `Dominant`, and `scent_embedding_v1`.
4. Inject a focus label, such as `Floral`.
5. Re-run the exact same input sequence with `M_Floral`.
6. Compare:
   - Does `Floral` remain visible?
   - Does the competing `Woods` evidence attenuate?
   - Does `Dominant` move toward `Floral` when appropriate?
   - Do adjacent short-lived labels remain plausible rather than disappearing completely?
   - Does clipping increase?

The expected result is not perfect single-label purity. A biological-style nose can still show adjacent transient notes. The goal is controlled directional influence.

## Metrics

Track at least:

- dominant-label duration before/after gain
- target-label active frames before/after gain
- competing-label active frames before/after gain
- false-positive no-scent frames before/after gain
- ADC clip count before/after gain
- embedding distance to target golden vector before/after gain
- embedding distance to competing golden vectors before/after gain

For embedding distance, start simple:

```text
cosine similarity(scent_embedding_v1, golden_label_embedding)
```

## UI Direction

The live UI should eventually expose:

- focus label selector
- mask on/off toggle
- raw vs masked run comparison
- applied mask display
- clipping warning
- residual readout view

The first implementation can be headless-only if that is faster:

```sh
scripts/run_headless_live_grid8.sh --focus Floral
```

or an equivalent explicit argument once the CLI shape is chosen.

## Clipping and Normalization Risks

The gain bridge must guard against focus artifacts:

- Boosted channels must hard-clip at the ADC maximum.
- Clip count must be reported.
- If clipping is frequent, the run should be treated as invalid or at least suspicious.
- Embedding output should remain normalized according to `scent_embedding_v1`; gain should not create arbitrary vector magnitude explosions.
- Pegged sensors are expected in real use. The first implementation should report clipping/pegging rather than trying to hide it; later mask strategies can decide whether saturated channels should be down-weighted, ignored, or treated as high-confidence evidence.

## Residual Bleed Risks

Attenuating one family may reveal or invent secondary labels.

That is useful diagnostic information, not automatically failure.

Classify residuals as:

- expected adjacent transient
- persistent competitor
- ghost caused by mask distortion
- clipping artifact
- model/readout artifact

## Near-Term Implementation Steps

1. Define a Rust `GainStage` module with identity and per-label masks.
2. Add a small mask config format under `data/gain/` or `config/gain/`.
3. Add gain-stage audit fields or a separate `gain_audit.csv`.
4. Add `--focus-label <label>` to `grid_live_headless` first.
5. Keep peak-pair support optional until Grid8 proves the validation loop.
6. Add golden-vector generation from current no-scent/single-note runs.
7. Add a comparison report:
   - no focus
   - focus label applied
   - target-vs-competitor embedding similarity changes
8. Add UI controls only after the headless validation loop is coherent.

## Open Questions

- Should v1 masks operate on raw ADC values, baseline-relative values, or quantized Grid8 bins?
- Should a focus label be allowed to boost channels above `1.0`, or should v1 only attenuate non-target channels?
- Should masks be manually authored first, derived from golden vectors, or both?
- Should focus intent persist for a fixed duration, until manually cleared, or only while a segment is active?
- Should the comparison target be `Dominant`, full 14-label output, `scent_embedding_v1`, or all three?
