# adamw.c

Pure, dependency-free C implementation of the **AdamW** optimizer.

AdamW = **Adam** with **decoupled weight decay** (Loshchilov & Hutter, 2019).

> The key insight: weight decay should shrink parameters *independently* of the gradient update — not be folded into the gradient as L2 regularization. Decoupling the two produces better-regularized models across the board.

---

## Algorithm

For a flat parameter segment of length `n`:

**1. Decoupled weight decay** (applied *before* the moment update)
```
params[i] ← params[i] · (1 − lr · wd)
```

**2. First and second moment updates** (same as Adam)
```
m[i] ← β₁·m[i] + (1−β₁)·g[i]
v[i] ← β₂·v[i] + (1−β₂)·g[i]²
```

**3. Bias correction**
```
m̂[i] = m[i] / (1 − β₁ᵗ)
v̂[i] = v[i] / (1 − β₂ᵗ)
```

**4. Parameter update**
```
params[i] -= lr · m̂[i] / (sqrt(v̂[i]) + ε)
```

The crucial difference from plain Adam (as used in `microgpt.c`) is step 1. In Adam, weight decay would modify the gradient: `g += wd · p`. In AdamW, weight decay is applied directly to the parameters before the gradient step. This means the effective regularization strength is independent of the adaptive per-parameter learning rate.

Moments are stored as `double` for numerical stability, matching `microgpt.c`'s existing `adam_m` / `adam_v` arrays.

---

## API

### `adamw_init`

```c
AdamWState *adamw_init(
    float **param_ptrs,  // pointers to the start of each segment in the flat params array
    float **grad_ptrs,   // pointers to the start of each segment in the flat grads  array
    int    *sizes,       // number of scalar floats in each segment
    int     n_segs,      // number of segments
    double  lr,          // learning rate          (default: 1e-2)
    double  beta1,       // first  moment decay    (default: 0.9 )
    double  beta2,       // second moment decay    (default: 0.95)
    double  eps,         // stability term         (default: 1e-8)
    double  weight_decay // decoupled weight decay (default: 0.0 )
);
```

Allocates per-segment first and second moment buffers (zero-initialised). All pointer arrays must remain valid for the lifetime of the returned state.

### `adamw_init_simple`

```c
AdamWState *adamw_init_simple(
    float  *params,
    float  *grads,
    int     n_params,
    double  lr, double beta1, double beta2, double eps, double weight_decay
);
```

Convenience wrapper for the common single-segment case — registers the entire flat `params` / `grads` arrays as one AdamW segment. Equivalent to calling `adamw_init` with `n_segs = 1`.

### `adamw_step`

```c
void adamw_step(AdamWState *s);
```

Performs one AdamW update across all registered segments. Increments the internal step counter (used for bias correction). Call after the backward pass, before zeroing gradients.

For LR scheduling, set `s->lr` before calling:
```c
s->lr = base_lr * (1.0 - (double)step / num_steps);
adamw_step(s);
```

### `adamw_free`

```c
void adamw_free(AdamWState *s);
```

Releases all heap memory owned by the state.

---

## Build

```bash
gcc -O2 -std=c11 -DADAMW_TEST -o adamw_test adamw.c -lm && ./adamw_test
```

`adamw.c` is self-contained — no headers to install, no dependencies beyond `libc` and `libm`. To use it in your project, copy the file and `#include "adamw.c"` or compile it alongside your source.

### Self-test output

```
=== adamw.c self-test ===

Test 1: Single step — params change
  Param[0] before: 1.000000, after: 0.990000
  Param changed: YES
  PASS

Test 2: Weight decay shrinks params with zero gradient
  Param norm before: 4.0000
  Param norm after 5 steps: 3.9800
  PASS

Test 3: Bias correction — step counter increments
  Step after 3 calls: 3
  PASS

Test 4: Multi-segment — independent moments per segment
  Segment 0 param[0] changed: YES
  Segment 1 param[0] changed: YES
  PASS

All tests passed.
```

---

## Integration with microgpt.c

### Drop-in replacement for plain Adam

`microgpt.c` currently uses plain Adam (no weight decay) with an inline loop. Replace it with `adamw_init_simple`:

**Before (microgpt.c current):**
```c
static double *adam_m, *adam_v;
adam_m = calloc(n_params, sizeof(double));
adam_v = calloc(n_params, sizeof(double));
double beta1 = 0.9, beta2 = 0.95, eps_adam = 1e-8;

// In loop:
double lr_t = learning_rate * (1.0 - (double)step / num_steps);
for (int i = 0; i < n_params; i++) {
    double g = grads[i];
    adam_m[i] = beta1 * adam_m[i] + (1 - beta1) * g;
    adam_v[i] = beta2 * adam_v[i] + (1 - beta2) * g * g;
    double m_hat = adam_m[i] / (1 - pow(beta1, step + 1));
    double v_hat = adam_v[i] / (1 - pow(beta2, step + 1));
    params[i] -= lr_t * m_hat / (sqrt(v_hat) + eps_adam);
}
```

**After (AdamW):**
```c
AdamWState *opt = adamw_init_simple(params, grads, n_params,
                                    learning_rate,
                                    0.9, 0.95, 1e-8,
                                    0.1); // weight_decay

// In loop:
opt->lr = learning_rate * (1.0 - (double)step / num_steps);
adamw_step(opt);
memset(grads, 0, n_params * sizeof(float));

// Cleanup:
adamw_free(opt);
```

### Hybrid with Muon (recommended)

Use AdamW for non-hidden parameters (embeddings, `lm_head`) and Muon for hidden weight matrices (`attn_wq/wk/wv/wo`, `mlp_fc1/fc2`). Multiple optimizer instances coexist cleanly — each owns its own moment buffers.

```c
// AdamW covers: wte, wpe, lm_head
float *adam_params[] = { params + offsets.wte,
                          params + offsets.wpe,
                          params + offsets.lm_head };
float *adam_grads[]  = { grads  + offsets.wte,
                          grads  + offsets.wpe,
                          grads  + offsets.lm_head };
int adam_sizes[] = { vocab_size * n_embd,
                     block_size * n_embd,
                     vocab_size * n_embd };

AdamWState *adam_opt = adamw_init(adam_params, adam_grads, adam_sizes, 3,
                                   3e-4, 0.9, 0.95, 1e-8, 0.1);

// Muon for hidden weights (see muon.c for full setup)
MuonState *muon_opt = muon_init(...);

// In training loop:
adam_opt->lr = adam_base_lr * (1.0 - (double)step / num_steps);
muon_opt->lr = muon_base_lr * (1.0 - (double)step / num_steps);
muon_step(muon_opt);
adamw_step(adam_opt);
memset(grads, 0, n_params * sizeof(float));
```

### Which parameters to use AdamW for

| Parameter | Optimizer |
|---|---|
| `wte` (token embeddings) | **AdamW** |
| `wpe` (position embeddings) | **AdamW** |
| `lm_head` | **AdamW** |
| `attn_wq`, `attn_wk`, `attn_wv`, `attn_wo` | Muon |
| `mlp_fc1`, `mlp_fc2` | Muon |

Embeddings and heads lack the 2-D matrix structure that Muon's Newton-Schulz orthogonalization targets, so AdamW is the right default for them.

---

## Hyperparameters

| Hyperparameter | Default | Notes |
|---|---|---|
| `lr` | `1e-2` | Tune per task; linear LR warmdown is common |
| `beta1` | `0.9` | First moment decay; rarely needs tuning |
| `beta2` | `0.95` | Second moment decay; `0.999` is the original Adam default |
| `eps` | `1e-8` | Stability; increase to `1e-6` for low-precision training |
| `weight_decay` | `0.0` | `0.1` is a common starting point for language models |

---

## Implementation notes

### Multi-segment design

A single `AdamWState` can manage multiple non-contiguous parameter slices, each with its own moment buffers and element count. This is useful when you want different learning rates or weight decay coefficients for different parameter groups — register them as separate segments, then set `opt->lr` before each step.

### Double-precision moments

Both `m` and `v` are stored as `double` arrays. In the early steps of training, `(1 − β₁ᵗ)` is small (e.g., `0.1` at step 1), so the bias-corrected estimate `m_hat = m / bc1` amplifies any floating-point error in `m`. Storing moments in `double` avoids catastrophic cancellation and matches `microgpt.c`'s existing convention.

### Weight decay short-circuit

When `weight_decay == 0.0`, the decay loop is skipped entirely — no multiply-by-one overhead. This makes `adamw_init_simple` a zero-cost drop-in when you don't need regularization.

### Step counter convention

The step counter is 1-indexed and managed internally, matching `microgpt.c`'s `(step + 1)` bias-correction convention. You never need to pass a step number — just call `adamw_step`.

---

## References

- Loshchilov & Hutter, ["Decoupled Weight Decay Regularization"](https://arxiv.org/abs/1711.05101) (2019) — the AdamW paper
- Kingma & Ba, ["Adam: A Method for Stochastic Optimization"](https://arxiv.org/abs/1412.6980) (2015) — original Adam
- [llm.c](https://github.com/karpathy/llm.c) — production C/CUDA ML reference; AdamW idioms
- [microgpt.c](https://github.com/loretoparisi/microgpt.c) — the Adam implementation this extends
- [muon.c](https://github.com/loretoparisi/muon.c) — sibling file; same structural conventions

---

## License

MIT
