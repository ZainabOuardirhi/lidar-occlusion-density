"""Density-aware proposal-level occlusion assessment."""
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score
from .or_score import fit_proposal_or_calibration_params, score_proposal_density_or_with_params

def split_by_sample_id(
    proposal_df,
    test_ratio=0.30,
    seed=42,
):
    """
    Split by sample_id so objects from the same image do not appear
    in both calibration and test sets.
    """
    rng = np.random.default_rng(seed)

    sample_ids = np.array(sorted(proposal_df["sample_id"].unique()))
    rng.shuffle(sample_ids)

    n_test = int(len(sample_ids) * test_ratio)

    test_ids = set(sample_ids[:n_test])
    calib_ids = set(sample_ids[n_test:])

    calib_df = proposal_df[
        proposal_df["sample_id"].isin(calib_ids)
    ].copy()

    test_df = proposal_df[
        proposal_df["sample_id"].isin(test_ids)
    ].copy()

    return calib_df, test_df, calib_ids, test_ids

def evaluate_or_thresholds_on_df(
    df,
    thresholds=np.arange(0, 101, 1),
    score_col="proposal_OR",
):
    rows = []

    y_true = df["gt_is_occluded"].values.astype(int)
    scores = df[score_col].values

    for threshold in thresholds:
        y_pred = (scores >= threshold).astype(int)

        tp = int(((y_true == 1) & (y_pred == 1)).sum())
        tn = int(((y_true == 0) & (y_pred == 0)).sum())
        fp = int(((y_true == 0) & (y_pred == 1)).sum())
        fn = int(((y_true == 1) & (y_pred == 0)).sum())

        tpr = tp / max(tp + fn, 1)
        fpr = fp / max(fp + tn, 1)
        tnr = tn / max(tn + fp, 1)

        balanced_accuracy = 0.5 * (tpr + tnr)

        rows.append({
            "threshold": threshold,
            "TPR": tpr,
            "FPR": fpr,
            "TNR": tnr,
            "BalancedAccuracy": balanced_accuracy,
            "TP": tp,
            "TN": tn,
            "FP": fp,
            "FN": fn,
        })

    return pd.DataFrame(rows)

def select_best_threshold(
    scored_calib_df,
):
    threshold_results = evaluate_or_thresholds_on_df(
        scored_calib_df,
    )

    best_row = (
        threshold_results
        .sort_values("BalancedAccuracy", ascending=False)
        .iloc[0]
    )

    return float(best_row["threshold"]), threshold_results

def evaluate_final_test_set(
    scored_df,
    threshold,
):
    y_true = scored_df["gt_is_occluded"].values.astype(int)
    y_score = scored_df["proposal_OR"].values
    y_pred = (y_score >= threshold).astype(int)

    auc = roc_auc_score(y_true, y_score)
    ap = average_precision_score(y_true, y_score)

    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    tpr = tp / max(tp + fn, 1)
    fpr = fp / max(fp + tn, 1)
    tnr = tn / max(tn + fp, 1)

    balanced_accuracy = 0.5 * (tpr + tnr)

    return {
        "AUC": auc,
        "AP": ap,
        "threshold": threshold,
        "TPR": tpr,
        "FPR": fpr,
        "TNR": tnr,
        "BalancedAccuracy": balanced_accuracy,
        "TP": tp,
        "TN": tn,
        "FP": fp,
        "FN": fn,
    }

def run_calibration_test_experiment(
    proposal_df,
    seed,
    test_ratio=0.30,
):
    calib_df, test_df, calib_ids, test_ids = split_by_sample_id(
        proposal_df,
        test_ratio=test_ratio,
        seed=seed,
    )

    or_params = fit_proposal_or_calibration_params(
        calib_df,
    )

    scored_calib = score_proposal_density_or_with_params(
        calib_df,
        params=or_params,
    )

    scored_test = score_proposal_density_or_with_params(
        test_df,
        params=or_params,
    )

    best_threshold, calib_threshold_results = select_best_threshold(
        scored_calib,
    )

    test_results = evaluate_final_test_set(
        scored_test,
        threshold=best_threshold,
    )

    test_results["seed"] = seed
    test_results["n_calib"] = len(scored_calib)
    test_results["n_test"] = len(scored_test)
    test_results["n_calib_frames"] = len(calib_ids)
    test_results["n_test_frames"] = len(test_ids)

    return test_results, scored_test
