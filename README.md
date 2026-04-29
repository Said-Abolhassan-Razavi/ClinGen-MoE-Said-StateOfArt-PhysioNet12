# ClinGen-MoE — PhysioNet 2012 SOTA Contribution

**Author:** Said Abolhassan Razavi  
**Project:** TER — Multimodal Mixture-of-Experts for Synthetic Clinical Data Generation  
**Université Paris-Saclay, Master 1 AI**  
**Supervisor:** Pr. Faicel Chamroukhi  
**Dataset:** PhysioNet Computing in Cardiology Challenge 2012

---

## Overview

This repository applies the three SOTA contributions to the **PhysioNet 2012** dataset — the project's primary evaluation benchmark (7,925 ICU patients, 48-hour stays).

It extends the MIMIC-IV state of the art (Said-StateOfArt branch) by:
- Working with **real PhysioNet 2012 data** (set-a: 4,000 patients)
- Handling **both modalities** jointly: static descriptors + time-series
- Using a **Conditional GRU-VAE** — the recommended model for this dataset scale

## Dataset

**PhysioNet Computing in Cardiology Challenge 2012**  
https://physionet.org/content/challenge-2012/1.0.0/

- 12,000 ICU records total (set-a + set-b + set-c)
- 4,000 patients in set-a (used here)
- **Two modalities:**
  - Static descriptors: Age, Gender, Height, Weight, ICUType
  - Time-series: 37 variables (vitals + labs) over 48 hours
- Outcome: In-hospital mortality, SAPS-I score, length of stay
- Freely available — no credentialed access required

## Data Setup

Place the PhysioNet 2012 dataset at:
```
data/predicting-mortality-of-icu-patients-the-physionet-computing-in-cardiology-challenge-2012-1.0.0/
├── set-a/          # 4,000 patient .txt files
├── set-b/          # 4,000 patient .txt files
├── Outcomes-a.txt
└── Outcomes-b.txt
```

Or update the `PHYSIONET_DIR` path at the top of the notebook.

## Notebook

### `01_physionet12_sota_contribution.ipynb`

Covers all three SOTA contributions applied to PhysioNet 2012:

| Section | Content |
|---|---|
| 1 — Load data | Parse real set-a patient files (both modalities) |
| 2 — Static descriptors | Age, Gender, ICUType distribution |
| 3 — Time-series | Feature coverage, missingness, temporal dynamics |
| 4 — Conditional GRU-VAE | Static features condition the GRU encoder |
| 5 — Five-axis evaluation | Fidelity, plausibility, temporal ACF, TSTR, DCR |
| 6 — Summary | Results table |

## SOTA Contributions Applied

### ① Clinical Language Models (ClinicalBERT)
Reviewed in the SOTA document. PhysioNet 2012 contains only structured time-series — clinical notes are the natural next modality to add (future work using ClinicalBERT).

### ② Sequence Models — Conditional GRU-VAE
**Recommended for PhysioNet 2012:** 7,925 patients, two modalities.

- GRU encoder processes time-series step-by-step (preserves autocorrelation)
- Static features (Age, Gender, ICUType) **condition** the encoder
- Missingness mask appended at each time step (missing ≠ zero)
- Cyclical KL annealing prevents posterior collapse

Why conditional over standard VRAE? A 75-year-old CCU patient has different physiological ranges than a 30-year-old SICU patient. Conditioning ensures generated trajectories are consistent with patient demographics.

### ③ Privacy Evaluation (DCR & NNDR)
DCR (Distance to Closest Record) computed in Section 5. Safe when DCR(synthetic) ≈ DCR(real held-out).

## Installation

```bash
pip install -r requirements.txt
```

## Running

```bash
jupyter notebook notebooks/
```
