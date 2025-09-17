# MAGIA – Gradient Inversion with Subset-Selection (Reference Implementation)

> **This implementation is derived from the paper**  
> **“MAGIA: SENSING PER-IMAGE SIGNALS FROM SINGLE-ROUND AVERAGED GRADIENTS FOR LABEL-INFERENCE-FREE GRADIENT INVERSION.”**  
> It realizes the optimization objective and the subset-selection schedule described in the paper’s main text and appendix.

---


## 1. Overview

This repository provides a research implementation of MAGIA for reconstructing multiple images from a single round of **averaged gradients** without relying on label inference. The code supports:
- Full-batch consistency matching and **random subset** consistency matching.
- A family of **selection schedules** for the subset size \(S\) (a.k.a. “selection mechanism”), matching the paper’s appendix.
- Standard image priors (e.g., TV) and common optimizers (e.g., L-BFGS/Adam).

The main training/attack loop is in **`main_MAGIA_sel.py`**.


---

## 2. Global Defaults and Hyperparameters

The script defines **global defaults** (e.g., number of iterations, TV regularization weight, seed, save paths) at module scope and/or as `argparse` defaults.
**You can adjust them in two ways:**

1. **Directly edit** the default values near the top of `main_MAGIA_sel.py` (module-level constants or variable assignments).
2. **Override via CLI flags** when invoking the script (recommended for reproducibility).

> Practical note: global defaults provide a sensible starting point; prefer CLI overrides in experiments to keep changes tracked in your run commands.

---

## 3. Selection Scheduling (Reproducing the Paper’s Appendix)

MAGIA alternates between a **full-batch term** and a **subset term** of size $S$. The **selection size $S$** is controlled by:

* `--sel_mode {const1,constk,linear_inc_l,linear_inc_total,linear_dec_l,linear_dec_total,sigmoid}`
* `--sel_k` (for `constk`)
* `--sel_l_iter` (linear warmup/decay length)
* `--sig_a, --sig_b` (sigmoid parameters)

All strategies are automatically **clamped** to $1 \le S \le B$.

### 3.1 Modes and Formulas

Let `iters` be the 0-based iteration index, `n_iter` the total number of iterations, and `batch_size = B`.

* **`const1`**
  $S = 1$ (maximally local; enhances per-image discrimination).

* **`constk`**
  $S = k$ (user-specified constant).
  Set via `--sel_k K`.

* **`linear_inc_l`** (linear increase over first `l` steps, then fixed to `B`)

  $$
  S =
  \begin{cases}
  \left\lceil B \cdot \frac{\text{iters}+1}{l} \right\rceil, & \text{iters} < l\\
  B, & \text{iters} \ge l
  \end{cases}
  $$

  Set `l` via `--sel_l_iter`.

* **`linear_inc_total`** (linear increase across total schedule)

  $$
  S = \left\lceil B \cdot \frac{\text{iters}+1}{n\_iter} \right\rceil
  $$

* **`linear_dec_l`** (linear decrease over first `l` steps, then fixed to 1)

  $$
  S =
  \begin{cases}
  \left\lceil B \cdot \left(1 - \frac{\text{iters}}{l}\right) \right\rceil, & \text{iters} < l\\
  1, & \text{iters} \ge l
  \end{cases}
  $$

  Set `l` via `--sel_l_iter`.

* **`linear_dec_total`** (linear decrease across total schedule)

  $$
  S = \left\lceil B \cdot \left(1 - \frac{\text{iters}}{n\_iter}\right) \right\rceil
  $$

* **`sigmoid`** (sigmoid-shaped schedule)

  $$
  S = \left\lceil B \cdot \text{sigmoid\_decay}(\text{iters}; a, b) \right\rceil,
  $$

  where `sigmoid_decay` is the same function referenced in the paper/code; set `a,b` via `--sig_a --sig_b`.

> These modes correspond one-to-one to the selection schedules listed in the appendix of the MAGIA paper (constant, linear warm-up/decay, total-horizon linear, and sigmoid schedule).

---

## 4. Command-Line Usage

Minimal run (defaults, equivalent to a small constant subset):

```bash
python main_MAGIA_sel.py --sel_mode constk --sel_k 2
```

Linear warm-up to full batch in 100 steps:

```bash
python main_MAGIA_sel.py --sel_mode linear_inc_l --sel_l_iter 100
```

Linear decay from full batch to single sample across total iterations:

```bash
python main_MAGIA_sel.py --sel_mode linear_dec_total
```

Sigmoid schedule (example parameters):

```bash
python main_MAGIA_sel.py --sel_mode sigmoid --sig_a -0.1 --sig_b 200
```

Adjust common hyperparameters (illustrative):

```bash
python main_MAGIA_sel.py \
  --sel_mode const1 \
  --n_iter 300 \
  --batch_size 32 \
  --tv_weight 5e-3 \
  --optimizer lbfgs
```

> Use `-h/--help` to view all available flags and defaults.

---

## 5. Reproducibility Notes

* Fix the random seed via the corresponding CLI flag or top-level default.
* Log the exact run command and git commit (if applicable).
* For L-BFGS runs, prefer saving intermediate checkpoints and the final reconstructed images; record the selection schedule and TV weight.

---

## 6. Citation

If you find this implementation useful, please cite the MAGIA paper:

```
MAGIA: Sensing Per-Image Signals from Single-Round Averaged Gradients for Label-Inference-Free Gradient Inversion.
```

---

## 7. License

This code is provided for academic research and reproducibility purposes.
Please check the repository’s license file (if present) before redistribution.
