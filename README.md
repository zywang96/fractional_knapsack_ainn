
# Fractional Knapsack AINN

This repository runs a full **3-stage AINN pipeline** for a 5-item fractional knapsack instance.  
You provide **weights** and **profits**, and the script produces:

- the **final fractional selection** (fractions for each item)
- the **final total profit**
- optional **monitoring signals** that help localize which stage likely went wrong

---

## What this repo does

### Stage 1 — Scoring
- Takes `(weights, profits)` as input.
- Predicts a 5-element score vector (ratio-like) used for sorting.

**Optional Stage1 curve monitor**
- Loads a fitted curve from a JSON file.
- For each item, compares:
  - curve-inferred value vs Stage1 output
- Flags any item whose difference exceeds the threshold.
- Flags the entire sequence if a comparative inconsistency is detected.

### Stage 2 — Ordering
- Produces the permutation (sorted indices) of the input ratio vector.
- Runs **Encoder** (monitor) to decide whether this Stage looks abnormal.

### Stage 3 — Selection
- Produces the final 5-element fraction vector and total profit.
- Runs **Encoder** (monitor) to decide whether this Stage looks abnormal.

### Optional Ground Truth
The script can also compute and print ground truth reference outputs:
- Stage1 GT: `p/w` ratios
- Stage2 GT: indices sorted by GT ratios
- Stage3 GT: optimal greedy fractional knapsack fractions

---

## How to use

### Basic single-instance run
```bash
python ainn_end2end_infer.py \
  --weights 2,5,4,7,1 \
  --profits 10,8,7,15,3
````

### Print ground-truth reference outputs

```bash
python ainn_end2end_infer.py \
  --weights 2,5,4,7,1 \
  --profits 10,8,7,15,3 \
  --print_gt
```

### Use a specific trained configuration / checkpoints

If you want to try your own model, you can use the `--config` flag to point the script to the appropriate checkpoints and settings.

```bash
python ainn_end2end_infer.py \
  --weights 2,5,4,7,1 \
  --profits 10,8,7,15,3 \
  --config ainn_less_iter
```

---

## Output overview

The script prints:

- Stage1 predicted scores (+ curve comparison flags if enabled)
- Stage2 predicted sorting indices + Encoder2 anomaly detection results
- Stage3 predicted selection + Encoder3 anomaly detection results
- Final fractions + total profit
- A highlighted **anomaly summary** (which stage(s) were flagged)
- Ground truth of given input


## Quick Start -- Colab

You can also try it on Colab by starting with this Jupyter [notebook](https://colab.research.google.com/drive/1eUlIePDL5l40NZiD55-kfDvqkGz--4yY?usp=sharing)
