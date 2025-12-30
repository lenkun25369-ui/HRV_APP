#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import os
import sys
import types

# ---- Force headless matplotlib ----
os.environ["MPLBACKEND"] = "Agg"

# ---- Completely disable biosppy plotting modules ----
fake_plotting = types.ModuleType("biosppy.inter_plotting")
sys.modules["biosppy.inter_plotting"] = fake_plotting
sys.modules["biosppy.inter_plotting.ecg"] = fake_plotting
sys.modules["biosppy.inter_plotting.acc"] = fake_plotting

# ---- Also block tkinter just in case ----
sys.modules["tkinter"] = None
sys.modules["_tkinter"] = None


import gc
import warnings
from collections import Counter

import numpy as np
import pandas as pd
import heartpy as hp
import neurokit2 as nk
import pyhrv
import pyhrv.tools as tools

from keras.models import load_model
from sklearn.preprocessing import StandardScaler

# ===== 必須存在於 repo 根目錄 =====
from function_all import (
    categorical_focal_loss,
    f1,
    wave_to_newtest_input_data_form1,
    normal_ecg_transfer,
    hrvtransform2_only_normal_ECG_filter_hr
)

# =========================
# 全域設定
# =========================
fs = 125
warnings.filterwarnings("ignore")
gc.enable()

# =========================
# ECG Grid Function
# =========================
def ECGgrid(data, timelag1=600, timelag2=1200, nrows=40, ncols=40):
    df = pd.DataFrame(data)
    ecg = df['ECG'].values
    ecg = StandardScaler().fit_transform(ecg.reshape(-1, 1)).flatten()

    def takens_embedding(x, tau):
        n = len(x) - tau
        return np.column_stack((x[:n], x[tau:tau+n]))

    def grid_counting(embedded):
        H, _, _ = np.histogram2d(
            embedded[:, 0], embedded[:, 1],
            bins=[nrows, ncols]
        )
        return (np.count_nonzero(H) / (nrows * ncols)) * 100

    g1 = grid_counting(takens_embedding(ecg, timelag1))
    g2 = grid_counting(takens_embedding(ecg, timelag2))

    return pd.DataFrame({
        "Grid Counting": [g1],
        "SgridTAU": [g2 - g1]
    })


# =========================
# 主流程
# =========================
def main(ecg_csv_path: str, out_csv_path: str):

    if not os.path.exists(ecg_csv_path):
        raise FileNotFoundError(f"ECG file not found: {ecg_csv_path}")

    # ---- Load model ----
    model = load_model(
        "models/model_focalloss.h5",
        custom_objects={
            "focal_loss": categorical_focal_loss(gamma=2),
            "f1": f1
        }
    )

    # ---- Read ECG ----
    df = pd.read_csv(ecg_csv_path)
    if "II" not in df.columns:
        raise ValueError("ECG CSV must contain column 'II'")

    II_data = df["II"].values
    del df

    hrv_update = hp.preprocessing.scale_data(II_data, lower=0, upper=200)

    # ---- Output header ----
    h0 = pd.DataFrame(columns=[
        'aFdP', 'fFdP', 'ARerr', 'DFA.Alpha.1',
        'Mean.rate', 'Poincar..SD2', 'shannEn',
        'LF.HF.ratio.LombScargle',
        'Grid Counting', 'SgridTAU', 'time_min'
    ])

    step0 = 37500
    v1 = len(hrv_update)
    bins = np.append(np.arange(0, v1, step0), v1)[1:]

    for i in bins:
        idx = np.arange(i - step0, i).astype(int)
        hrdata = hrv_update[idx]
        hrdata = hrdata[~np.isnan(hrdata)]

        if len(hrdata) < step0:
            continue

        # ---- DL classification ----
        x1, x2 = wave_to_newtest_input_data_form1(hrdata, show=False)
        preds = np.argmax(
            model.predict([x1, x2], verbose=0),
            axis=-1
        )

        data_ecg = normal_ecg_transfer(hrdata, preds, fs)

        # ---- HRV features ----
        hrv_part = hrvtransform2_only_normal_ECG_filter_hr(
            hrdata, preds, fs,
            settings_time=None,
            settings_welch=None,
            settings_ar=None,
            settings_lomb=None,
            settings_nonlinear=None
        )

        grid_part = ECGgrid(data_ecg)
        hrv_part = pd.concat([hrv_part.reset_index(drop=True),
                              grid_part.reset_index(drop=True)], axis=1)

        hrv_part["time_min"] = i / 7500
        h0 = pd.concat([h0, hrv_part], ignore_index=True)

    if h0.empty:
        raise RuntimeError("No HRV features generated")

    h0.to_csv(out_csv_path, index=False)
    print(f"[OK] HRV features saved to {out_csv_path}")


# =========================
# CLI Entry
# =========================
if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python generate_hrv_10_features.py <ECG_5min.csv> <h0.csv>")
        sys.exit(2)

    main(sys.argv[1], sys.argv[2])
