#ifndef ADAMW_C
#define ADAMW_C

/*
 * adamw.c
 *
 * AdamW optimizer — pure, dependency-free C implementation.
 *
 * AdamW = Adam with **decoupled** weight decay (Loshchilov & Hutter, 2019).
 *
 * The key difference from plain Adam (as used in microgpt.c) is that weight
 * decay is applied directly to the parameters *before* the gradient update,
 * rather than being folded into the gradient.  This decoupling means weight
 * decay acts as a true L2 regularizer regardless of the adaptive learning
 * rate, which produces better-regularized models in practice.
 *
 * Plain Adam (microgpt.c current):
 *   p -= lr * m_hat / (sqrt(v_hat) + eps)
 *   (weight decay, if any, would modify the gradient: g += wd * p)
 *
 * AdamW (this file):
 *   p  ← p * (1 − lr * wd)                   [decoupled weight decay]
 *   p  ← p − lr * m_hat / (sqrt(v_hat) + eps) [Adam moment update]
 *
 * Algorithm for a flat parameter segment of length `n`:
 *
 *   Given: params[n], grads[n], first moment m[n], second moment v[n], step t
 *
 *   1. Weight decay (decoupled, applied before moment update):
 *        params[i] *= (1 − lr * wd)
 *
 *   2. Moment updates (same as Adam):
 *        m[i] = β₁·m[i] + (1−β₁)·g[i]
 *        v[i] = β₂·v[i] + (1−β₂)·g[i]²
 *
 *   3. Bias correction:
 *        m̂[i] = m[i] / (1 − β₁ᵗ)
 *        v̂[i] = v[i] / (1 − β₂ᵗ)
 *
 *   4. Parameter update:
 *        params[i] -= lr * m̂[i] / (sqrt(v̂[i]) + ε)
 *
 * Defaults (matching microgpt.c's existing Adam and common practice):
 *   β₁ = 0.9, β₂ = 0.95, ε = 1e-8, wd = 0.0
 *
 * First and second moments are stored as double for numerical stability,
 * matching microgpt.c's existing `adam_m` / `adam_v` arrays.
 *
 * Design mirrors muon.c: flat pointer API, heap-allocated state struct,
 * init / step / free lifecycle.  Multiple AdamW instances can coexist
 * (e.g., one for embeddings + lm_head, alongside a Muon instance for
 * hidden weight matrices).
 *
 * Usage:
 *   // Register a contiguous slice of the flat params / grads arrays:
 *   AdamWState *opt = adamw_init(params + offset, grads + offset, n,
 *                                lr, beta1, beta2, eps, weight_decay);
 *   // In training loop, after backward:
 *   adamw_step(opt);
 *   memset(grads + offset, 0, n * sizeof(float));
 *   // Cleanup:
 *   adamw_free(opt);
 *
 * Build & self-test:
 *   gcc -O2 -std=c11 -DADAMW_TEST -o adamw_test adamw.c -lm && ./adamw_test
 *
 * References:
 *   Loshchilov & Hutter, "Decoupled Weight Decay Regularization" (2019)
 *   https://arxiv.org/abs/1711.05101
 *   microgpt.c — Adam implementation this extends
 *   muon.c     — sibling file; same structural conventions
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

/* =========================================================================
 * AdamWParam — state for a single contiguous parameter segment
 *
 * A "segment" maps to one or more weight matrices stored contiguously in
 * the flat params array.  In practice you can register the entire params
 * array as one segment (n = n_params), or split into sub-ranges (e.g.,
 * separate segments for embeddings vs lm_head for different lr values).
 * ========================================================================= */
typedef struct {
    float  *param;   /* pointer into the flat params array                     */
    float  *grad;    /* pointer into the flat grads  array                     */
    double *m;       /* first  moment (EMA of gradients),      length = n      */
    double *v;       /* second moment (EMA of squared grads),  length = n      */
    int     n;       /* number of scalar parameters in this segment            */
} AdamWParam;

/* =========================================================================
 * AdamWState — optimizer state for all AdamW-managed parameter segments
 * ========================================================================= */
typedef struct {
    AdamWParam *segments;    /* array of per-segment states                    */
    int         n_segments;  /* number of registered segments                  */
    double      lr;          /* learning rate            (default: 1e-2)       */
    double      beta1;       /* first  moment decay      (default: 0.9 )       */
    double      beta2;       /* second moment decay      (default: 0.95)       */
    double      eps;         /* numerical stability term (default: 1e-8)       */
    double      weight_decay;/* decoupled weight decay   (default: 0.0 )       */
    int         step;        /* global step counter, 1-indexed (for bias corr) */
} AdamWState;

/* =========================================================================
 * adamw_init — create an AdamWState managing `n_segs` parameter segments.
 *
 * param_ptrs[i]  — pointer to the start of segment i in the flat params array
 * grad_ptrs[i]   — pointer to the start of segment i in the flat grads  array
 * sizes[i]       — number of scalar floats in segment i
 * n_segs         — number of segments
 * lr             — learning rate
 * beta1          — first  moment exponential decay rate (default 0.9)
 * beta2          — second moment exponential decay rate (default 0.95)
 * eps            — denominator stability term           (default 1e-8)
 * weight_decay   — decoupled weight decay coefficient   (default 0.0)
 *
 * All pointer arrays must remain valid for the lifetime of the state.
 * Returns a heap-allocated AdamWState; release with adamw_free().
 * ========================================================================= */
AdamWState *adamw_init(float **param_ptrs, float **grad_ptrs, int *sizes,
                       int n_segs,
                       double lr, double beta1, double beta2,
                       double eps, double weight_decay) {
    AdamWState *s = (AdamWState *)malloc(sizeof(AdamWState));
    if (!s) { fprintf(stderr, "adamw: OOM allocating state\n"); exit(1); }

    s->segments    = (AdamWParam *)malloc(n_segs * sizeof(AdamWParam));
    if (!s->segments) { fprintf(stderr, "adamw: OOM allocating segments\n"); exit(1); }

    s->n_segments  = n_segs;
    s->lr          = lr;
    s->beta1       = beta1;
    s->beta2       = beta2;
    s->eps         = eps;
    s->weight_decay= weight_decay;
    s->step        = 0;

    for (int i = 0; i < n_segs; i++) {
        AdamWParam *seg = &s->segments[i];
        seg->param = param_ptrs[i];
        seg->grad  = grad_ptrs[i];
        seg->n     = sizes[i];

        /* Zero-initialise moments (calloc guarantees 0.0 for IEEE doubles) */
        seg->m = (double *)calloc(sizes[i], sizeof(double));
        seg->v = (double *)calloc(sizes[i], sizeof(double));
        if (!seg->m || !seg->v) {
            fprintf(stderr, "adamw: OOM allocating moments for segment %d "
                            "(%d params)\n", i, sizes[i]);
            exit(1);
        }
    }

    return s;
}

/* =========================================================================
 * adamw_init_simple — convenience wrapper for the common single-segment case.
 *
 * Registers the entire flat params/grads arrays as one AdamW segment.
 * Equivalent to:
 *   adamw_init(&params, &grads, &n_params, 1, lr, beta1, beta2, eps, wd)
 *
 * This maps directly onto microgpt.c's current usage pattern where all
 * n_params scalars are updated with a single optimizer.
 * ========================================================================= */
AdamWState *adamw_init_simple(float *params, float *grads, int n_params,
                              double lr, double beta1, double beta2,
                              double eps, double weight_decay) {
    return adamw_init(&params, &grads, &n_params, 1,
                      lr, beta1, beta2, eps, weight_decay);
}

/* =========================================================================
 * adamw_step — perform one AdamW update across all registered segments.
 *
 * Increments the internal step counter, computes bias-correction factors,
 * and applies the four-step AdamW update to every scalar parameter.
 *
 * Caller workflow:
 *   1. Forward pass  → compute loss.
 *   2. Backward pass → populate grad arrays.
 *   3. adamw_step()  → update params, increment step.
 *   4. Zero grad arrays for the next iteration.
 *
 * The step counter is managed internally.  If you need LR scheduling
 * (e.g., the linear LR warmdown in microgpt.c), set s->lr before each call:
 *   s->lr = base_lr * (1.0 - (double)step / num_steps);
 *   adamw_step(s);
 * ========================================================================= */
void adamw_step(AdamWState *s) {
    s->step += 1;   /* 1-indexed, matching microgpt.c's (step + 1) convention */

    /* Bias-correction denominators — computed once per step, shared across
     * all segments and all parameters within a segment.                      */
    double bc1 = 1.0 - pow(s->beta1, s->step);   /* 1 − β₁ᵗ */
    double bc2 = 1.0 - pow(s->beta2, s->step);   /* 1 − β₂ᵗ */

    for (int si = 0; si < s->n_segments; si++) {
        AdamWParam *seg = &s->segments[si];
        float  *param = seg->param;
        float  *grad  = seg->grad;
        double *m     = seg->m;
        double *v     = seg->v;
        int     n     = seg->n;

        /* ---------------------------------------------------------------
         * Step 1: Decoupled weight decay
         *   W ← W · (1 − lr · wd)
         *
         * Applied BEFORE the moment update, as per the AdamW paper.
         * When weight_decay == 0.0 this is a no-op and the branch is
         * skipped for efficiency.
         * --------------------------------------------------------------- */
        if (s->weight_decay != 0.0) {
            double wd_factor = 1.0 - s->lr * s->weight_decay;
            for (int i = 0; i < n; i++) {
                param[i] = (float)((double)param[i] * wd_factor);
            }
        }

        /* ---------------------------------------------------------------
         * Steps 2–4: Adam moment update + bias-corrected parameter step
         *
         * m[i] ← β₁·m[i] + (1−β₁)·g             (first  moment EMA)
         * v[i] ← β₂·v[i] + (1−β₂)·g²            (second moment EMA)
         * m̂    ← m[i] / (1 − β₁ᵗ)               (bias correction)
         * v̂    ← v[i] / (1 − β₂ᵗ)               (bias correction)
         * param[i] -= lr · m̂ / (sqrt(v̂) + ε)    (parameter update)
         *
         * We use double precision for moments (matching microgpt.c) to
         * avoid catastrophic cancellation in the early steps where
         * (1 − β₁ᵗ) is very small.
         * --------------------------------------------------------------- */
        double one_minus_b1 = 1.0 - s->beta1;
        double one_minus_b2 = 1.0 - s->beta2;

        for (int i = 0; i < n; i++) {
            double g = (double)grad[i];

            /* Update moments */
            m[i] = s->beta1 * m[i] + one_minus_b1 * g;
            v[i] = s->beta2 * v[i] + one_minus_b2 * g * g;

            /* Bias-corrected estimates */
            double m_hat = m[i] / bc1;
            double v_hat = v[i] / bc2;

            /* Parameter update */
            param[i] -= (float)(s->lr * m_hat / (sqrt(v_hat) + s->eps));
        }
    }
}

/* =========================================================================
 * adamw_free — release all heap memory owned by an AdamWState.
 * ========================================================================= */
void adamw_free(AdamWState *s) {
    for (int i = 0; i < s->n_segments; i++) {
        free(s->segments[i].m);
        free(s->segments[i].v);
    }
    free(s->segments);
    free(s);
}

/* =========================================================================
 * Integration guide for microgpt.c
 * =========================================================================
 *
 * CURRENT microgpt.c (plain Adam, no weight decay):
 * --------------------------------------------------
 *   static double *adam_m;
 *   static double *adam_v;
 *   adam_m = calloc(n_params, sizeof(double));
 *   adam_v = calloc(n_params, sizeof(double));
 *   double beta1 = 0.9, beta2 = 0.95, eps_adam = 1e-8;
 *
 *   // In loop:
 *   double lr_t = learning_rate * (1.0 - (double)step / num_steps);
 *   for (int i = 0; i < n_params; i++) {
 *       double g = grads[i];
 *       adam_m[i] = beta1 * adam_m[i] + (1 - beta1) * g;
 *       adam_v[i] = beta2 * adam_v[i] + (1 - beta2) * g * g;
 *       double m_hat = adam_m[i] / (1 - pow(beta1, step + 1));
 *       double v_hat = adam_v[i] / (1 - pow(beta2, step + 1));
 *       params[i] -= lr_t * m_hat / (sqrt(v_hat) + eps_adam);
 *   }
 *
 * REPLACEMENT (AdamW, all params in one segment):
 * ------------------------------------------------
 *   AdamWState *opt = adamw_init_simple(params, grads, n_params,
 *                                       learning_rate,
 *                                       0.9, 0.95, 1e-8,
 *                                       0.1); // weight_decay
 *
 *   // In loop (LR scheduling):
 *   opt->lr = learning_rate * (1.0 - (double)step / num_steps);
 *   adamw_step(opt);
 *   memset(grads, 0, n_params * sizeof(float));
 *
 *   // Cleanup:
 *   adamw_free(opt);
 *
 * HYBRID with Muon (recommended):
 * --------------------------------
 * Use AdamW for non-hidden parameters (embeddings, lm_head) and Muon
 * for hidden weight matrices (attn_wq/wk/wv/wo, mlp_fc1/fc2).
 *
 *   // Adam segment covers: wte, wpe, lm_head
 *   // These are contiguous in microgpt.c's layout: offsets.wte ... lm_head
 *   int adam_offsets[] = { offsets.wte, offsets.wpe, offsets.lm_head };
 *   int adam_sizes[]   = { vocab_size * n_embd,
 *                          block_size * n_embd,
 *                          vocab_size * n_embd };
 *   float *adam_params[] = { params + adam_offsets[0],
 *                             params + adam_offsets[1],
 *                             params + adam_offsets[2] };
 *   float *adam_grads[]  = { grads  + adam_offsets[0],
 *                             grads  + adam_offsets[1],
 *                             grads  + adam_offsets[2] };
 *   AdamWState *adam_opt = adamw_init(adam_params, adam_grads, adam_sizes, 3,
 *                                     3e-4, 0.9, 0.95, 1e-8, 0.1);
 *
 *   // Muon for hidden weights (see muon.c for full setup)
 *   MuonState *muon_opt = muon_init(...);
 *
 *   // In training loop:
 *   adam_opt->lr = adam_base_lr * (1.0 - (double)step / num_steps);
 *   muon_opt->lr = muon_base_lr * (1.0 - (double)step / num_steps);
 *   muon_step(muon_opt);
 *   adamw_step(adam_opt);
 *   memset(grads, 0, n_params * sizeof(float));
 *
 * =========================================================================
 *
 * Self-test: compile with -DADAMW_TEST to run sanity checks.
 *
 * ========================================================================= */
#endif /* ADAMW_C */
