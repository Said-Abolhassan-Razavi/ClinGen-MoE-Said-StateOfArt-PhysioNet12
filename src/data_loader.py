"""
data_loader.py
--------------
Load and preprocess PhysioNet 2012 patient records.

Handles both modalities:
  - Modality 1: Static descriptors (Age, Gender, ICUType)
  - Modality 2: Time-series measurements (24 clinical features, 48-hour window)

Feature schema matches the MIMIC-IV 24-feature architecture so models
can be compared across both datasets.

Author: Said Abolhassan Razavi
Project: TER, Université Paris-Saclay, Master 1 AI
"""

import glob
import os
from collections import defaultdict

import numpy as np
import pandas as pd

# ── Feature schema — 24 features matching MIMIC-IV architecture ──────────────
ALL_FEATURES = [
    "Heart_Rate", "Systolic_BP", "Diastolic_BP", "Mean_BP",
    "Respiratory_Rate", "O2_Saturation", "GCS_Verbal", "GCS_Eye", "GCS_Motor",
    "Glucose", "Potassium", "Sodium", "Chloride", "Creatinine",
    "Urea_Nitrogen", "Bicarbonate", "Anion_Gap", "Hemoglobin",
    "Hematocrit", "Magnesium", "Platelet_Count", "Phosphate",
    "White_Blood_Cells", "Calcium_Total",
]
N_FEAT   = len(ALL_FEATURES)
FEAT_IDX = {name: i for i, name in enumerate(ALL_FEATURES)}

# PhysioNet parameter → MIMIC-IV feature name
_PARAM_MAP = {
    "HR": "Heart_Rate",
    "NISysABP": "Systolic_BP", "SysABP": "Systolic_BP",
    "NIDiasABP": "Diastolic_BP", "DiasABP": "Diastolic_BP",
    "NIMAP": "Mean_BP", "MAP": "Mean_BP",
    "RespRate": "Respiratory_Rate",
    "SaO2": "O2_Saturation",
    "Glucose": "Glucose", "K": "Potassium", "Na": "Sodium",
    "Creatinine": "Creatinine", "BUN": "Urea_Nitrogen",
    "HCO3": "Bicarbonate", "HCT": "Hematocrit",
    "Mg": "Magnesium", "Platelets": "Platelet_Count",
    "WBC": "White_Blood_Cells",
}
# Note: Chloride, Anion_Gap, Phosphate, Calcium_Total not in PhysioNet 2012
# → those feature indices are always unobserved (mask=0)

ICU_NAMES = {1: "CCU", 2: "CSRU", 3: "MICU", 4: "SICU"}


def _parse_patient(fpath: str) -> dict | None:
    """
    Parse a single PhysioNet 2012 patient .txt file.

    Returns a dict with keys:
        id        : int — RecordID
        static    : dict — Age, Gender, ICUType (np.nan if missing)
        ts_val    : (T, 24) float32 — observed values
        ts_mask   : (T, 24) float32 — 1 where observed, 0 where missing
        ts_times  : (T,)    float32 — hours since admission
    Returns None if the patient has fewer than 3 distinct timestamps.
    """
    static = {"Age": np.nan, "Gender": np.nan, "ICUType": np.nan}
    ts_buf: dict = defaultdict(dict)   # time_hrs -> {feat_idx: value}
    record_id = None

    with open(fpath) as f:
        for line in f:
            parts = line.strip().split(",", 2)
            if len(parts) != 3 or parts[0] == "Time":
                continue
            t_str, param, val_str = parts
            try:
                val = float(val_str)
            except ValueError:
                continue
            if val < -0.5:          # PhysioNet sentinel for missing
                continue

            # ── Static features at time 00:00 ────────────────────────────────
            if t_str == "00:00":
                if param == "RecordID":
                    record_id = int(val)
                elif param in ("Age", "Gender", "ICUType"):
                    static[param] = val
                continue

            # ── Time-series rows ─────────────────────────────────────────────
            hh, mm = t_str.split(":")
            t_hrs = int(hh) + int(mm) / 60.0
            if t_hrs > 48.0:
                continue

            if param == "GCS":
                excess = max(0.0, min(12.0, val - 3.0))
                for fname, fval in [
                    ("GCS_Eye",    1.0 + excess * 3.0 / 12.0),
                    ("GCS_Verbal", 1.0 + excess * 4.0 / 12.0),
                    ("GCS_Motor",  1.0 + excess * 5.0 / 12.0),
                ]:
                    fi = FEAT_IDX[fname]
                    if fi not in ts_buf[t_hrs]:
                        ts_buf[t_hrs][fi] = fval

            elif param == "HCT":
                ts_buf[t_hrs][FEAT_IDX["Hematocrit"]] = val
                if FEAT_IDX["Hemoglobin"] not in ts_buf[t_hrs]:
                    ts_buf[t_hrs][FEAT_IDX["Hemoglobin"]] = val / 3.0   # HGB ≈ HCT/3

            elif param in _PARAM_MAP:
                fi = FEAT_IDX[_PARAM_MAP[param]]
                if fi not in ts_buf[t_hrs]:   # non-invasive < invasive priority
                    ts_buf[t_hrs][fi] = val

    times = sorted(ts_buf.keys())
    if len(times) < 3:
        return None

    T = len(times)
    ts_val  = np.zeros((T, N_FEAT), dtype=np.float32)
    ts_mask = np.zeros((T, N_FEAT), dtype=np.float32)
    for t, thr in enumerate(times):
        for fi, fv in ts_buf[thr].items():
            ts_val[t, fi]  = fv
            ts_mask[t, fi] = 1.0

    return {
        "id":       record_id,
        "static":   static,
        "ts_val":   ts_val,
        "ts_mask":  ts_mask,
        "ts_times": np.array(times, dtype=np.float32),
    }


def make_static_vector(static: dict) -> np.ndarray:
    """
    Encode static features as a 7-dimensional float vector:
        [age_normalised, gender_oh_0, gender_oh_1,
         icu_oh_0, icu_oh_1, icu_oh_2, icu_oh_3]

    Suitable as conditioning input to the Conditional GRU-VAE.
    """
    age  = float(static["Age"]) if not np.isnan(static["Age"]) else 60.0
    age_norm = (age - 60.0) / 15.0

    g    = int(static["Gender"])  if not np.isnan(static["Gender"])  else 0
    g_oh = [1, 0] if g == 0 else [0, 1]

    icu  = int(static["ICUType"]) if not np.isnan(static["ICUType"]) else 1
    icu_oh = [0, 0, 0, 0]
    if 1 <= icu <= 4:
        icu_oh[icu - 1] = 1

    return np.array([age_norm] + g_oh + icu_oh, dtype=np.float32)


def pad_to_length(ts_val: np.ndarray, ts_mask: np.ndarray,
                  T_max: int = 48) -> tuple[np.ndarray, np.ndarray]:
    """Pad or truncate a patient's time-series to exactly T_max timesteps."""
    T = ts_val.shape[0]
    if T >= T_max:
        return ts_val[:T_max], ts_mask[:T_max]
    pad = np.zeros((T_max - T, N_FEAT), dtype=np.float32)
    return np.vstack([ts_val, pad]), np.vstack([ts_mask, pad])


def load_physionet_set(physionet_dir: str,
                       subset: str = "set-a",
                       T_max: int = 48,
                       min_timestamps: int = 3) -> list[dict]:
    """
    Load all patient records from a PhysioNet 2012 subset.

    Parameters
    ----------
    physionet_dir : str
        Path to the PhysioNet 2012 challenge directory
        (containing set-a/, set-b/, Outcomes-a.txt, etc.)
    subset : str
        Which subset to load: 'set-a', 'set-b', or 'both'
    T_max : int
        Pad/truncate all time-series to this length (default 48 hours)
    min_timestamps : int
        Skip patients with fewer distinct timestamps

    Returns
    -------
    list of dicts, each with keys:
        id, static, ts_val (T_max,24), ts_mask (T_max,24),
        ts_times (T_max,), static_vec (7,), label
    """
    # Load outcome labels
    labels = {}
    for fname in ("Outcomes-a.txt", "Outcomes-b.txt"):
        fpath = os.path.join(physionet_dir, fname)
        if not os.path.isfile(fpath):
            continue
        df = pd.read_csv(fpath)
        for _, row in df.iterrows():
            labels[int(row["RecordID"])] = int(row["In-hospital_death"])

    if not labels:
        raise FileNotFoundError(
            f"No Outcomes-*.txt found in {physionet_dir}")

    # Collect patient files
    subsets = ["set-a", "set-b"] if subset == "both" else [subset]
    all_files = []
    for s in subsets:
        d = os.path.join(physionet_dir, s)
        if os.path.isdir(d):
            all_files.extend(sorted(glob.glob(os.path.join(d, "*.txt"))))

    records = []
    for fpath in all_files:
        rec = _parse_patient(fpath)
        if rec is None:
            continue
        if rec["id"] not in labels:
            continue
        if len(rec["ts_times"]) < min_timestamps:
            continue

        v, m = pad_to_length(rec["ts_val"], rec["ts_mask"], T_max)
        t    = np.pad(rec["ts_times"], (0, max(0, T_max - len(rec["ts_times"]))),
                      constant_values=0.0)[:T_max]

        records.append({
            "id":          rec["id"],
            "static":      rec["static"],
            "static_vec":  make_static_vector(rec["static"]),
            "ts_val":      v,
            "ts_mask":     m,
            "ts_times":    t.astype(np.float32),
            "label":       labels[rec["id"]],
        })

    print(f"Loaded {len(records)} patients from {subset}")
    n_pos = sum(r["label"] for r in records)
    print(f"  Mortality: {n_pos}/{len(records)} ({100*n_pos/len(records):.1f}%)")
    return records
