"""
src/classifier/predict.py

Prediction module for intelligibility classifier.
Handles data preprocessing, model loading, prediction, and saving results.

This module serves only for prediction tasks. Training is handled in src/classifier/train.py.
"""

from omegaconf import DictConfig
import numpy as np
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from src.classifier.utils import preprocess_csv, data_split, save_metrics
import joblib

logger = logging.getLogger(__name__)


def load_models(config: DictConfig) -> Dict[str, object]:
    """
    Load pre-trained models based on config.

    :param config: DictConfig object.
    :return: Dictionary of model name to loaded model.
    """
    models = {}

    pth = Path(config.retrained_models_path)
    if config.get("use_retrained", False):
        if not pth.exists():
            logger.warning("Retrained models path %s does not exist. Falling back to pretrained models.", Path(config.retrained_models_path).resolve())
            pth = Path(config.pretrained_models_path)
    else:
        pth = Path(config.pretrained_models_path)

    for model_cfg in config.model_type:
        model_name = model_cfg._target_.split('.')[-1]
        try:
            if model_name.lower() == "catboostclassifier":
                model_path = pth / f"{model_name}.cbm"
                from catboost import CatBoostClassifier
                model = CatBoostClassifier()
                model.load_model(model_path)
            else:
                model_path = pth / f"{model_name}.joblib"
                model = joblib.load(model_path)
            models[model_name] = model
            logger.info("Loaded model: %s from %s", model_name, model_path)
        except Exception as e:
            logger.error("Error loading model %s from %s: %s", model_name, model_path, e)
            raise
    return models


def apply_scaler(X: np.ndarray, config: DictConfig) -> np.ndarray:
    """
    Apply scaler to features based on config.

    :param X: Features to scale.
    :param config: DictConfig object.
    :return: Scaled features.
    """
    scale_features = config.preprocessing.scaler.get("enabled", False)

    if not scale_features:
        logger.info("Feature scaling is disabled in config.")
        return X
    
    if config.get("use_retrained", False):
        scaler_path = Path(config.retrained_models_path).joinpath("scaler.joblib").resolve()
        if scaler_path.exists():
            try:
                scaler = joblib.load(scaler_path)
                logger.info("Loaded retrained scaler from %s", scaler_path)
            except Exception as e:
                logger.error("Error loading retrained scaler from %s: %s", scaler_path, e)
                raise
        else:
            logger.warning("Retrained scaler path %s does not exist. Falling back to pretrained scaler.", scaler_path)
            scaler_path = Path(config.pretrained_models_path) / "scaler.joblib"
            scaler = joblib.load(scaler_path)
            logger.info("Loaded pretrained scaler from %s", scaler_path)

    else:
        scaler_path = Path(config.pretrained_models_path).joinpath("scaler.joblib").resolve()
        try:
            scaler = joblib.load(scaler_path)
            logger.info("Loaded pretrained scaler from %s", scaler_path)
        except Exception as e:
            logger.error("Error loading pretrained scaler from %s: %s", scaler_path, e)
            raise
            
    try:
        X_scaled = scaler.fit_transform(X)
        logger.info("Applied scaler: %s", config.preprocessing.scaler.source)
    except Exception as e:
        logger.error("Error applying scaler %s: %s", config.preprocessing.scaler.source, e)
        raise
    return X_scaled
    

def predict(X: np.ndarray, models: Dict[str, object], config: DictConfig) -> Dict[str, np.ndarray]:

    """
    Predict using loaded models.

    :param X: Features for prediction.
    :param models: Dictionary of model name to loaded model.
    :return: Dictionary of model name to prediction probabilities.
    """
    y_pred_dict = {}
    X_scaled = apply_scaler(X, config) 
    for model_name, model in models.items():
        try:
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_scaled)[:, 1]  # Probability of positive class
            else:
                y_pred_proba = model.decision_function(X_scaled)
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())

            y_pred_dict[model_name] = y_pred_proba
            logger.info("Trained and predicted with model: %s", model_name)
        except Exception as e:
            logger.error("Error training/predicting with model %s: %s", model_name, e)
            raise
    return y_pred_dict
    

def binarize_target(y: np.ndarray, threshold: float) -> np.ndarray:
    """
    Binarize target values based on threshold.

    :param y: Continuous target values.
    :param threshold: Threshold for binarization.
    :return: Binarized target array.
    """
    return (y < threshold).astype(int)


def prepare_data(config: DictConfig, mode: str='prediction') -> tuple:
    """
    Preprocess CSV and return features, targets, corpus, and sample IDs.

    :param config: DictConfig object.
    :return: tuple (X, y, corpus, sample_id, y_bin)
    """
    X, y, corpus, sample_id, transcriptions = preprocess_csv.run(config, mode=mode)
    X = X.to_numpy()
    y = y.to_numpy() if y is not None else None
    corpus = corpus.to_numpy() if corpus is not None else None
    sample_id = sample_id.to_numpy() if sample_id is not None else None

    if y is not None:
        class_threshold = config.get("class_threshold", 0.3)
        y_bin = binarize_target(y, class_threshold)
    else:
        y_bin = None

    return X, y, corpus, sample_id, y_bin, transcriptions


def save_results(config: DictConfig, sample_id: np.ndarray, y_pred_dict: Dict[str, np.ndarray], texts: Union[pd.Series, None]=None):
    """
    Save prediction results to CSV.

    :param config: DictConfig object.
    :param sample_id: Array of sample IDs.
    :param y_pred_dict: Dictionary of model name to prediction probabilities.
    :param texts: Series of text transcription. None, if it is not provided.
    """
    post_hoc_threshold = config.get("post_hoc_threshold", 0.7)

    df_samples = pd.DataFrame({'sample_id': sample_id})

    stacked_probs = np.column_stack(list(y_pred_dict.values()))
    avg_stacked_probs = np.mean(stacked_probs, axis=1)
    df_samples['y_pred_prob_ensemble'] = avg_stacked_probs
    df_samples['y_pred_class_ensemble'] = (avg_stacked_probs >= post_hoc_threshold).astype(int)

    for model_name, y_pred_probs in y_pred_dict.items():
        y_pred_bin = (y_pred_probs >= config.get("post_hoc_threshold", 0.7)).astype(int)
        df_samples[f'y_pred_prob_{model_name}'] = y_pred_probs
        df_samples[f'y_pred_class_{model_name}'] = y_pred_bin
    if texts is not None:
        df_samples = df_samples.merge(
            texts[['file', 'text_large']],
            left_on='sample_id',
            right_on='file',
            how='left'
        )
        df_samples.drop(columns='file', inplace=True)
        df_samples.rename(columns={'text_large': 'text'}, inplace=True)
        df_samples = df_samples[['sample_id', 'text'] + [c for c in df_samples.columns if c not in ['sample_id', 'text']]]

    Path(config.prediction_output_path).mkdir(parents=True, exist_ok=True)
    samples_path = Path(config.prediction_output_path).joinpath("predictions.csv").resolve()
    df_samples.to_csv(samples_path, index=False)
    logger.info("Saved predictions to %s", samples_path)

    
def run(config: DictConfig, transcribe_mode: bool=True):
    """
    Main prediction entrypoint.
    - Preprocesses data
    - Loads models
    - Makes predictions
    - Evaluates and saves metrics if true labels are available

    :param config: DictConfig object.
    """
    models = load_models(config)
    if transcribe_mode:
        config['evaluate'] = False
        X, _, _, sample_id, _, transcriptions = prepare_data(config, mode='transcription')
    else:
        X, y, corpus, sample_id, y_bin, _ = prepare_data(config)
    y_pred_dict = predict(X, models, config)

    if config.get("evaluate", False) and y is not None:
        output_data = {}
        if corpus is not None:
            unique_corpora = np.unique(corpus)
            for corp in unique_corpora:
                mask = corpus == corp
                output_data[corp] = {
                    "sample_id_test": sample_id[mask],
                    "y_test": y[mask],
                    "y_test_bin": y_bin[mask],
                    "y_pred_dict": {k: v[mask] for k, v in y_pred_dict.items()}
                }
        else:
            output_data["dataset"] = {
                "sample_id_test": sample_id,
                "y_test": y,
                "y_test_bin": y_bin,
                "y_pred_dict": y_pred_dict,
            }

        save_metrics.run(output_data, config)
    else:
        logger.info("No true labels provided or evaluation disabled; skipping evaluation.")

    # Separately save prediction results irrespective of evaluation
    if transcribe_mode:
        save_results(config, sample_id, y_pred_dict, transcriptions)
    else:
        save_results(config, sample_id, y_pred_dict)
    logger.info("Prediction process completed.")


