# NoseKnows Scent Embedding

`scent_embedding_v1` is the first fixed-width bridge from the live Rust model output to downstream systems such as NoseLLM.

It is intentionally not a human-facing classifier. The fragrance wheel and `Dominant` row are display readouts. The embedding is a richer state vector that preserves current model output, recent history, coactivation, and model-specific features.

## Contract

```text
version: scent_embedding_v1
dims:    1024
type:    f32
format:  pipe-delimited vector in CSV artifacts
```

The live headless runners emit one embedding per input frame:

```text
data/live/embeddings.csv        peak-pair live model
data/live/grid_embeddings.csv   grid8 rolling model
```

Each row includes:

```text
run_id,row_index,elapsed_ms,stream_segment,source_sample_id,embedding_version,dims,vector
```

## Label Order

All label-aligned blocks use the 14-slice Edwards wheel order:

```text
0  Floral
1  Soft Floral
2  Floral Amber
3  Amber
4  Soft Amber
5  Woody Amber
6  Woods
7  Mossy Woods
8  Dry Woods
9  Aromatic
10 Citrus
11 Water
12 Green
13 Fruity
```

## Dimension Map

The first 256 dimensions are model-output and readout history. Dimensions `256..511` carry model-specific features. Dimensions `512..1023` are reserved for later learned projections or richer state.

```text
000..013 current logits, tanh-scaled
014..027 current positive logit strength, clipped and scaled to 0..1
028..041 short-window mean logits, tanh-scaled
042..055 short-window max logits, tanh-scaled
056..069 long-window mean logits, tanh-scaled
070..083 long-window max logits, tanh-scaled
084..097 long-window top-3 frequency per label
098..111 long-window top-1 frequency per label
112..125 top-3 recency per label, 1.0 means seen this frame
126..139 current top-3 flags
140..153 current top-1 flag
154..244 long-window pairwise top-3 coactivation, 14 choose 2 label pairs
245..255 reserved
256..511 model-specific feature prefix
512..1023 reserved
```

Window names are implementation constants in `src/embedding.rs`:

```text
short window = 8 frames
long window  = 32 frames
```

At the current live sample period of 100 ms, these correspond to roughly 0.8 seconds and 3.2 seconds. They are frame windows, not wall-clock guarantees.

## Model-Specific Feature Prefix

`256..511` is intentionally model-specific:

- Peak-pair live model: normalized peak/pair feature vector prefix.
- Grid8 rolling model: the 64 normalized `8 sensors x 8 one-second lookback` grid cells, followed by zeros.

Downstream code should check `embedding_version`, model/run provenance, and the result artifact that produced the embedding before assuming semantics for this block.

## Usage Guidance

Use the embedding when downstream logic needs more than one label:

- scent-state retrieval
- prompt/context construction for NoseLLM
- comparing exposures over time
- detecting persistent versus transient adjacent notes
- learning a higher-level classifier over model history

Do not use `scent_embedding_v1` as a replacement for raw results during debugging. Keep the companion `model_results.csv` or `grid_model_results.csv` when inspecting why a display readout made a decision.
