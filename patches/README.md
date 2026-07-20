# Preserved patches (not applied)

Optimizations that are kept here for future use but **deliberately NOT wired into
the shipped code** — the active `modeling_bolmo.py` is the original/faithful loop
so evals run the canonical implementation with zero divergence risk.

## `hf-static-shape-decode.patch`

Static-shape rewrite of the HF Bolmo generate loop
(`src/olmo_core/nn/bolmo/hf/modeling_bolmo.py`). Makes the whole decode step
static-shape (full-batch + masked `torch.where` state write-back instead of the
variable-row `selective_get`/`selective_put`), which:

- removes the per-step `MaskState` `.cpu()/.item()` syncs → **+23–27% eager**
  throughput at bs8/bs16 (bit-identical greedy output — verified via matching
  gsm8k scores and generation hashes);
- unblocks `torch.compile(model.model)` (with
  `torch._dynamo.config.allow_unspec_int_on_nn_module = True`) → bs1 1.86×
  (still faithful), bs16 1.58× over original.

**Why it is shelved, not shipped:** on real olmes eval the +26% pure-decode gain
dilutes to ~6% wall (prefill + per-step stop-sequence decoding + harness overhead
dominate), and we prefer the untouched canonical loop for eval fidelity. Full
analysis + measurements: `omnistudent/.claude/progress/05-bolmo-generation-speed.md`.

### To plug it back in

```bash
git apply patches/hf-static-shape-decode.patch
```

The full optimized implementation also lives in git history at commits
`653a9b7` (token-append vectorization + static encoder/xLSTM) and `26426de`
(static local decoder + full-batch loop reconcile) — base for the patch is
`653a9b7~1`. If the patch ever fails to apply (e.g. the released modeling
diverges), recover the exact code from those commits.
