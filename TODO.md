# NoseKnows TODO

This file is intentionally short. Use:

- `PLAN-001.md` for live injection and streaming model boundary.
- `PLAN-002.md` for manual gain/attenuation bridge validation.
- `ROADMAP.md` for larger future work streams.
- `NOTEBOOK.md` for historical findings, checkpoints, and research notes.

## Active Next Steps

1. Validate the Grid8 gain stage zero-input invariant against a known no-focus run.
2. Run focused Grid8 comparisons for a small set of mixed sequences:
   - `Floral + Woods` with `--focus-label Floral`
   - `Citrus + Woods` with `--focus-label Citrus`
   - `Mossy Woods + Floral` with `--focus-label Mossy Woods`
3. Add a small comparison report for no-focus vs focus runs:
   - dominant-label duration
   - target-label active frames
   - competing-label active frames
   - false-positive no-scent frames
   - clip count from `grid_gain_audit.csv`
4. Generate first-pass golden vectors for no-scent and the 14 fragrance labels from the current no-scent/single-note Grid8 runs.
5. Add cosine similarity against golden vectors to the focus comparison report.
6. Decide whether first-pass masks should remain attenuate-only or allow limited gain above `1.0`.
7. Keep the live UI centered on Grid8, dominant readout, and `scent_embedding_v1` while this focus validation is underway.
8. Begin real hardware captures once the bench sensor package is ready.

## Guardrails

- No focus label means the gain stage is identity and Grid8 behavior should match the previous no-focus path.
- Do not move gain logic into Python/Daft or UI code.
- Do not bury gain logic inside `grid_live_headless`; keep it in reusable Rust modules.
- Keep generated runtime artifacts ignored by git.
- Treat synthetic data as a pipeline and representation test, not real fragrance truth.
