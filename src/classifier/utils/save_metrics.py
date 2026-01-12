"""
src/classifier/utils/save_metrics.py

Handles evaluation metrics computation and storage.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from sklearn.metrics import (
    precision_recall_curve,
    auc,
    precision_score,
    recall_score,
    f1_score,
    accuracy_score,
)
from numpy import ndarray
from omegaconf import DictConfig
import warnings

logger = logging.getLogger(__name__)

METRICS = ['auc', 'precision', 'recall', 'f1', 'accuracy']
LARGE_VALUE = 1e9 # to avoid problems when a prediction class has no samples

def binarize_prediction(y_pred: ndarray, threshold: float) -> ndarray:
    """
    Binarize prediction probabilities based on threshold.

    :param y_pred: Array of prediction probabilities.
    :param threshold: Threshold for binarization.
    :return: Binarized prediction array.
    """
    return (y_pred >= threshold).astype(int)

def compute_metrics(y_true: ndarray,
                    y_true_bin: ndarray,
                    y_pred_probs: ndarray,
                    y_pred_bin: ndarray,
                    model_name: str,
                    config: DictConfig) -> dict:
    """
    Compute evaluation metrics for predictions.

    :param y_true: True continuous target values.
    :param y_true_bin: True binarized target values.
    :param y_pred_probs: Predicted probabilities.
    :param y_pred_bin: Binarized predictions.
    :param model_name: Name of the model.
    :param config: DictConfig object with evaluation settings.
    :return: Dictionary of computed metrics.
    """
    metrics_score = {}
    evaluation_metrics = set(config.get("evaluation_metrics", METRICS))
    warnings.filterwarnings("ignore", category=UserWarning)

    for metric in METRICS:
        if metric not in evaluation_metrics:
            continue
        try:
            if metric == "auc":
                precisions, recalls, _ = precision_recall_curve(y_true_bin, y_pred_probs)
                metrics_score["auc"] = auc(recalls, precisions)
            elif metric == "precision":
                metrics_score["precision"] = precision_score(y_true_bin, y_pred_bin, zero_division=0)
            elif metric == "recall":
                metrics_score["recall"] = recall_score(y_true_bin, y_pred_bin, zero_division=0)
            elif metric == "f1":
                metrics_score["f1"] = f1_score(y_true_bin, y_pred_bin, zero_division=0)
            elif metric == "accuracy":
                metrics_score["accuracy"] = accuracy_score(y_true_bin, y_pred_bin)
        except ValueError as e:
            logger.warning("%s computation failed for %s: %s", metric, model_name, e)
            metrics_score[metric] = LARGE_VALUE
        
    # Mean and median WER for positive predictions
    if np.any(y_pred_bin == 1):
        metrics_score["mean_low_wer"] = np.mean(y_true[y_pred_bin == 1])
        metrics_score["median_low_wer"] = np.median(y_true[y_pred_bin == 1])
    else:
        metrics_score["mean_low_wer"] = LARGE_VALUE
        metrics_score["median_low_wer"] = LARGE_VALUE

    return metrics_score

def save_per_sample_predictions(split_name: str, data: dict, config: DictConfig) -> pd.DataFrame:
    """
    Save per-sample predictions to CSV. It also includes average predictions across models (ensemble).

    :param split_name: Name of the data split (e.g., corpus name or fold number).
    :param data: Dictionary containing sample IDs, true labels, and predictions.
    :param config: DictConfig object with output settings.
    :return: DataFrame of per-sample predictions.
    """
    sample_id = data.get('sample_id_test', None)
    y_true = data['y_test']
    y_true_bin = data['y_test_bin']
    y_pred_dict = data['y_pred_dict']
    post_hoc_threshold = config.get("post_hoc_threshold", 0.7)
    
    df_samples = pd.DataFrame({
        'sample_id': sample_id,
        'true WER': y_true,
        'true class': y_true_bin,
    })

    ensemble_probs = np.mean(np.column_stack(list(y_pred_dict.values())), axis=1)
    ensemble_bin = binarize_prediction(ensemble_probs, post_hoc_threshold)
    df_samples["y_pred_prob_ensemble"] = ensemble_probs
    df_samples["y_pred_class_ensemble"] = ensemble_bin

    for model_name, y_pred_probs in y_pred_dict.items():
        y_pred_bin = binarize_prediction(y_pred_probs, post_hoc_threshold)
        df_samples[f"y_pred_prob_{model_name}"] = y_pred_probs
        df_samples[f"y_pred_class_{model_name}"] = y_pred_bin

    Path(config.evaluation_output_path).mkdir(parents=True, exist_ok=True)
    samples_path = Path(config.evaluation_output_path) / f"samples_{split_name}.csv"
    df_samples.to_csv(samples_path, index=False)
    logger.info("Saved per-sample predictions to %s", samples_path)

    return df_samples
                                
def save_metrics_summary(all_metrics: dict, config: DictConfig):
    """
    Saves summary metrics for each model to CSV files.
    
    :param all_metrics: Dictionary of each data split mapping model names to their metrics.
                       : It is structured as {split_name: {model_name: {metric_name: value}}}
    :param config: DictConfig object with output settings.
    """
    Path(config.evaluation_output_path).mkdir(parents=True, exist_ok=True)

    models = set()
    for split_metrics in all_metrics.values():
        models.update(split_metrics.keys())

    for model_name in models:
        records = [] # List of dicts for DataFrame
        for split_name, model_metrics in all_metrics.items():
            if model_name in model_metrics:
                record = {'split': split_name}
                record.update(model_metrics[model_name])
                records.append(record)
        df_summary = pd.DataFrame(records)

        if not df_summary.empty:
            avg_row = {'split': 'average'}
            for metric in df_summary.columns[1:]:
                avg_row[metric] = df_summary[metric].mean()
            df_summary = pd.concat([df_summary, pd.DataFrame([avg_row])], ignore_index=True)
            summary_path = Path(config.evaluation_output_path) / f"summary_{model_name}.csv"
            df_summary.to_csv(summary_path, index=False)
            logger.info("Saved summary metrics to %s", summary_path)
        else:
            logger.warning("No metrics recorded for model %s; skipping summary file.", model_name)

def run(output_data: dict, config: DictConfig):
    """
    Main entrypoint to save metrics and per-sample predictions.

    :param output_data: Dictionary containing data splits with sample IDs, true labels, and predictions.
    :param config: DictConfig object with output settings.
    """
    all_metrics = {} # {split_name: {model_name: {metric_name: value}}}

    for split_name, data in output_data.items():
        y_true = data['y_test']
        y_true_bin = data['y_test_bin']
        y_pred_dict = data['y_pred_dict']
        post_hoc_threshold = config.get("post_hoc_threshold", 0.7)

        model_metrics = {}

        for model_name, y_pred_probs in y_pred_dict.items():
            y_pred_bin = binarize_prediction(y_pred_probs, post_hoc_threshold)
            metrics_score = compute_metrics(
                y_true,
                y_true_bin,
                y_pred_probs,
                y_pred_bin,
                model_name,
                config
            )
            model_metrics[model_name] = metrics_score

        # Ensemble metrics
        ensemble_probs = np.mean(np.column_stack(list(y_pred_dict.values())), axis=1)
        ensemble_bin = binarize_prediction(ensemble_probs, post_hoc_threshold)
        model_metrics["ensemble"] = compute_metrics(y_true, y_true_bin, ensemble_probs, ensemble_bin, "ensemble", config)

        all_metrics[split_name] = model_metrics

        # Save per-sample predictions for this split
        save_per_sample_predictions(split_name, data, config)

    # Save summary metrics
    save_metrics_summary(all_metrics, config)
    logger.info("All metrics and per-sample predictions have been saved.")

def save_csv(df_samples: pd.DataFrame, path: str):
    df_samples.to_csv(path, index=False)
