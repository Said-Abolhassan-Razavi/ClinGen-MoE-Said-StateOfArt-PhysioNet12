"""
evaluation.py
-------------
Five-axis evaluation framework for synthetic clinical data.

Axes:
  1. Statistical Fidelity  — Wasserstein distance, KS test
  2. Clinical Plausibility — physiological range checks
  3. Temporal Fidelity     — autocorrelation ACF-1
  4. Utility (TSTR)        — Train-on-Synthetic Test-on-Real AUROC
  5. Privacy               — DCR (Distance to Closest Record) & NNDR

SOTA motivation:
  TSTR measures utility but not privacy. A model that memorises training
  patients can score well on TSTR while leaking real patient information.
  DCR and NNDR provide dedicated privacy measurement.

  Safe condition: DCR(synthetic) ≈ DCR(real held-out)

Author: Said Abolhassan Razavi
Project: TER, Université Paris-Saclay, Master 1 AI
"""

import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, pairwise_distances
from sklearn.preprocessing import StandardScaler

from .data_loader import ALL_FEATURES, N_FEAT

# ── Physiological plausibility bounds ────────────────────────────────────────
PHYSIO_BOUNDS = {
    "Heart_Rate":        (20,  300),
    "Systolic_BP":       (50,  250),
    "Diastolic_BP":      (20,  150),
    "Mean_BP":           (30,  200),
    "Respiratory_Rate":  (4,   60),
    "O2_Saturation":     (50,  100),
    "GCS_Verbal":        (1,   5),
    "GCS_Eye":           (1,   4),
    "GCS_Motor":         (1,   6),
    "Glucose":           (20,  700),
    "Creatinine":        (0.1, 25),
    "Sodium":            (100, 180),
    "Potassium":         (1.5, 10),
    "Hematocrit":        (10,  65),
    "White_Blood_Cells": (0.5, 100),
}


def _patient_mean_features(ts_arr: np.ndarray,
                           mask_arr: np.ndarray,
                           stat_arr: np.ndarray) -> np.ndarray:
    """Build feature matrix: mean observed value per feature + static."""
    rows = []
    for i in range(len(ts_arr)):
        ts_means = []
        for fi in range(N_FEAT):
            obs = ts_arr[i, :, fi][mask_arr[i, :, fi] == 1]
            ts_means.append(obs.mean() if len(obs) > 0 else 0.0)
        rows.append(ts_means + stat_arr[i].tolist())
    return np.array(rows, dtype=np.float32)


# ── Axis 1: Statistical Fidelity ─────────────────────────────────────────────
def compute_fidelity(real_ts: np.ndarray, real_mask: np.ndarray,
                     synth_ts: np.ndarray) -> dict:
    """
    Compute Wasserstein distance and KS test p-value per feature.

    Parameters
    ----------
    real_ts   : (N, T, F)
    real_mask : (N, T, F)
    synth_ts  : (N, T, F)

    Returns
    -------
    dict with keys: wasserstein (list), ks_pval (list), feature_names (list)
    """
    wass, ks_pvals = [], []
    for fi in range(N_FEAT):
        real_obs = []
        for i in range(len(real_ts)):
            obs = real_ts[i, :, fi][real_mask[i, :, fi] == 1]
            real_obs.extend(obs.tolist())
        synth_obs = synth_ts[:, :, fi].flatten().tolist()
        if len(real_obs) > 10:
            wass.append(wasserstein_distance(real_obs, synth_obs))
            _, pval = ks_2samp(real_obs, synth_obs)
            ks_pvals.append(pval)
        else:
            wass.append(np.nan)
            ks_pvals.append(np.nan)

    valid_wass = [w for w in wass if not np.isnan(w)]
    ks_pass = sum(1 for p in ks_pvals if not np.isnan(p) and p > 0.05)
    ks_total = sum(1 for p in ks_pvals if not np.isnan(p))

    print(f"  Mean Wasserstein: {np.mean(valid_wass):.3f}")
    print(f"  KS pass (p>0.05): {ks_pass}/{ks_total} features")
    return {"wasserstein": wass, "ks_pval": ks_pvals,
            "feature_names": ALL_FEATURES}


# ── Axis 2: Clinical Plausibility ────────────────────────────────────────────
def check_plausibility(synth_ts: np.ndarray) -> dict:
    """Check whether synthetic values fall within physiological bounds."""
    results = {}
    for fname, (lo, hi) in PHYSIO_BOUNDS.items():
        fi = ALL_FEATURES.index(fname)
        vals = synth_ts[:, :, fi].flatten()
        pct_valid = 100 * np.mean((vals >= lo) & (vals <= hi))
        results[fname] = {"range": (lo, hi), "pct_valid": pct_valid}
        status = "✅" if pct_valid >= 90 else "⚠"
        print(f"  {status} {fname:>25s}: {pct_valid:.1f}% in [{lo}, {hi}]")
    return results


# ── Axis 3: Temporal Fidelity — ACF-1 ────────────────────────────────────────
def acf1(series: list) -> float:
    """Autocorrelation at lag 1."""
    s = np.array(series)
    if len(s) < 3 or s.std() < 1e-8:
        return np.nan
    s = s - s.mean()
    return float(np.corrcoef(s[:-1], s[1:])[0, 1])


def compute_temporal_fidelity(real_ts: np.ndarray, real_mask: np.ndarray,
                              synth_ts: np.ndarray,
                              feature: str = "Heart_Rate") -> dict:
    """
    Compute ACF-1 for real and synthetic data on a given feature.

    ACF-1 benchmark:
        Real ICU HR:    ≈ 0.85  (highly autocorrelated)
        Vanilla VAE:    ≈ -0.05 (destroys temporal structure)
        Cond. GRU-VAE:  target > 0.40
    """
    fi = ALL_FEATURES.index(feature)
    real_acfs = []
    for i in range(len(real_ts)):
        obs = real_ts[i, :, fi][real_mask[i, :, fi] == 1].tolist()
        if len(obs) >= 3:
            real_acfs.append(acf1(obs))

    synth_acfs = [acf1(synth_ts[i, :, fi].tolist())
                  for i in range(len(synth_ts))]
    synth_acfs = [a for a in synth_acfs if not np.isnan(a)]

    r_mean = float(np.nanmean(real_acfs))
    s_mean = float(np.nanmean(synth_acfs))
    print(f"  Real  ACF-1 ({feature}): {r_mean:.3f}")
    print(f"  Synth ACF-1 ({feature}): {s_mean:.3f}")
    print(f"  (Vanilla VAE baseline ≈ -0.05 | target ≥ 0.40)")
    return {"real_acf1": r_mean, "synth_acf1": s_mean}


# ── Axis 4: Utility — TSTR ───────────────────────────────────────────────────
def compute_tstr(real_ts: np.ndarray, real_mask: np.ndarray,
                 real_stat: np.ndarray, real_labels: np.ndarray,
                 synth_ts: np.ndarray, synth_stat: np.ndarray) -> dict:
    """
    Train-on-Synthetic Test-on-Real evaluation.

    Trains a logistic regression classifier on synthetic data,
    tests it on real held-out data. Compares AUROC to TRTR baseline.

    TSTR ratio ≥ 0.85 → synthetic data has good utility.
    """
    X_real  = _patient_mean_features(real_ts,  real_mask,
                                     real_stat)
    X_synth = _patient_mean_features(synth_ts,
                                     np.ones_like(synth_ts), synth_stat)

    split   = int(0.7 * len(X_real))
    scaler  = StandardScaler().fit(X_real[:split])
    Xr_tr   = scaler.transform(X_real[:split])
    Xr_te   = scaler.transform(X_real[split:])
    Xs_tr   = StandardScaler().fit_transform(X_synth[:split])
    yr_tr   = real_labels[:split]
    yr_te   = real_labels[split:]

    if len(np.unique(yr_tr)) < 2 or len(np.unique(yr_te)) < 2:
        print("  Not enough class diversity for AUROC.")
        return {}

    clf_trtr = LogisticRegression(max_iter=500, random_state=42).fit(Xr_tr, yr_tr)
    clf_tstr = LogisticRegression(max_iter=500, random_state=42).fit(Xs_tr, yr_tr)

    auroc_trtr = roc_auc_score(yr_te, clf_trtr.predict_proba(Xr_te)[:, 1])
    auroc_tstr = roc_auc_score(yr_te,
                               clf_tstr.predict_proba(scaler.transform(X_real[split:]))[:, 1])
    ratio = auroc_tstr / auroc_trtr if auroc_trtr > 0 else np.nan

    print(f"  TRTR AUROC (real → real):      {auroc_trtr:.3f}")
    print(f"  TSTR AUROC (synthetic → real): {auroc_tstr:.3f}")
    print(f"  TSTR ratio:                    {ratio:.3f}  (target ≥ 0.85)")
    return {"auroc_trtr": auroc_trtr, "auroc_tstr": auroc_tstr, "ratio": ratio}


# ── Axis 5: Privacy — DCR & NNDR ─────────────────────────────────────────────
def compute_dcr_nndr(real_ts: np.ndarray, real_mask: np.ndarray,
                     real_stat: np.ndarray,
                     synth_ts: np.ndarray, synth_stat: np.ndarray,
                     n_sample: int = 200) -> dict:
    """
    Compute DCR and NNDR privacy metrics.

    DCR  — Distance to Closest Record:
        For each synthetic patient, the minimum distance to any real
        training patient. Concentrated near zero → memorisation risk.

    NNDR — Nearest Neighbour Distance Ratio:
        DCR normalised by distance to 2nd nearest real patient.
        More robust to feature-scale differences across modalities.

    Safe condition: DCR(synthetic) ≈ DCR(real held-out)
    If synthetic patients are no closer to training data than
    real held-out patients, the model generalised rather than memorised.

    Parameters
    ----------
    n_sample : int — number of patients to use (for speed)
    """
    X_real  = _patient_mean_features(real_ts,  real_mask,  real_stat)
    X_synth = _patient_mean_features(synth_ts,
                                     np.ones_like(synth_ts), synth_stat)

    n = min(n_sample, len(X_real))
    scaler = StandardScaler().fit(X_real)
    Xr = scaler.transform(X_real)[:n]
    Xs = scaler.transform(X_synth)[:n]

    # Split real into train / held-out
    split = n // 2
    Xr_train, Xr_held = Xr[:split], Xr[split:]
    Xs_sample = Xs[:split]

    D_synth = pairwise_distances(Xs_sample, Xr_train)
    D_real  = pairwise_distances(Xr_held,  Xr_train)

    dcr_synth = D_synth.min(axis=1)
    dcr_real  = D_real.min(axis=1)

    def _nndr(D):
        s = np.sort(D, axis=1)
        return s[:, 0] / (s[:, 1] + 1e-8)

    nndr_synth = _nndr(D_synth)
    nndr_real  = _nndr(D_real)

    gap = abs(dcr_synth.mean() - dcr_real.mean())
    verdict = "✅ Safe" if gap < 1.0 else "⚠ Gap detected"

    print(f"  DCR  — Synthetic:   {dcr_synth.mean():.3f} ± {dcr_synth.std():.3f}")
    print(f"  DCR  — Real held-out: {dcr_real.mean():.3f} ± {dcr_real.std():.3f}")
    print(f"  NNDR — Synthetic:   {nndr_synth.mean():.3f}")
    print(f"  NNDR — Real held-out: {nndr_real.mean():.3f}")
    print(f"  Gap |DCR(synth) - DCR(real)| = {gap:.3f}  → {verdict}")

    return {
        "dcr_synth":  dcr_synth,  "dcr_real":   dcr_real,
        "nndr_synth": nndr_synth, "nndr_real":  nndr_real,
        "gap": gap, "verdict": verdict,
    }
