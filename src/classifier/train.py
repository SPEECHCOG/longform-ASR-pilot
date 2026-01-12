"""
src/classifier/train.py

Training module for intelligibility classifier.
Handles data preprocessing, model training, evaluation, and saving.
"""

import hydra
from omegaconf import DictConfig
from numpy import ndarray
import logging
from pathlib import Path
from typing import Dict, Any, Optional, Union
from src.classifier.utils import preprocess_csv, data_split, save_metrics
import joblib

logger = logging.getLogger(__name__)

def instantiate_models(config: DictConfig) -> Dict[str, object]:
    """
    Instantiate models based on config.

    :param config: DictConfig object.
    :return: Dictionary of model name to instantiated model.
    """
    models = {}
    for model_cfg in config.model_type:
        model_name = model_cfg._target_.split('.')[-1]
        try:
            model = hydra.utils.instantiate(model_cfg)
            models[model_name] = model
            logger.info("Instantiated model: %s", model_name)
        except Exception as e:
            logger.error("Error instantiating model %s: %s", model_name, e)
            raise
    return models

def apply_scaler(X_train: ndarray, X_test: ndarray, config: DictConfig) -> tuple:
    """
    Apply scaler to training and test data.

    :param X_train: Training features.
    :param X_test: Test features.
    :param config: DictConfig object.
    :return: tuple (X_train_scaled, X_test_scaled, scaler)
    """
    scale_features = config.preprocessing.scaler.get("enabled", False)

    if not scale_features:
        logger.info("Feature scaling is disabled in config.")
        return X_train, X_test, None
    
    scaler = hydra.utils.instantiate(config.preprocessing.scaler.source)
    try:
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        logger.info("Applied scaler: %s", config.preprocessing.scaler.source)
    except Exception as e:
        logger.error("Error applying scaler %s: %s", config.preprocessing.scaler.source, e)
        raise
    return X_train_scaled, X_test_scaled, scaler
    
def train_and_predict(X_train: ndarray, 
                      y_train: ndarray,
                      X_test: ndarray, 
                      models: Dict[str, object]) -> Dict[str, ndarray]:

    """
    Train each model and get prediction probabilities.

    :param X_train: Training features.
    :param y_train: Training binary targets.
    :param X_test: Test features.
    :param models: Dictionary of model name to instantiated model.
    :return: Dictionary of model name to prediction probabilities.
    """
    y_pred_dict = {}
    for model_name, model in models.items():
        try:
            model.fit(X_train, y_train)
            if hasattr(model, "predict_proba"):
                y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
            else:
                y_pred_proba = model.decision_function(X_test)
                y_pred_proba = (y_pred_proba - y_pred_proba.min()) / (y_pred_proba.max() - y_pred_proba.min())

            y_pred_dict[model_name] = y_pred_proba
            logger.info("Trained and predicted with model: %s", model_name)
        except Exception as e:
            logger.error("Error training/predicting with model %s: %s", model_name, e)
            raise
    return y_pred_dict
    
def binarize_target(y: ndarray, threshold: float) -> ndarray:
    """
    Binarize target values based on threshold.

    :param y: Continuous target values.
    :param threshold: Threshold for binarization.
    :return: Binarized target array.
    """
    return (y < threshold).astype(int)

def prepare_data(config: DictConfig) -> tuple:
    """
    Preprocess CSV and return features, targets, corpus, and sample IDs.

    :param config: DictConfig object.
    :return: tuple (X, y, corpus, sample_id, y_bin)
    """
    X, y, corpus, sample_id, _ = preprocess_csv.run(config, mode="training")
    X = X.to_numpy()
    y = y.to_numpy()
    corpus = corpus.to_numpy() if corpus is not None else None
    sample_id = sample_id.to_numpy() if sample_id is not None else None

    class_threshold = config.get("class_threshold", 0.3)
    y_bin = binarize_target(y, class_threshold)

    return X, y, corpus, sample_id, y_bin

def apply_training(X_train: ndarray,
                   y_train: ndarray,
                   X_test: ndarray,
                   y_test: ndarray,
                   corpus_test: Optional[ndarray],
                   sample_id_test: Optional[ndarray],
                   config: DictConfig,
                   fold_idx: int) -> Dict[str, Any]: # -> or Dict[str, Union[ndarray, Dict[str, ndarray]]]
    """
    Apply scaling, instantiate models, train and predict.

    Returns a dictionary with:
        - sample_id_test: file names of test samples
        - y_test: original target values of test samples
        - y_test_bin: binarized target values of test samples
        - y_pred_dict: dict of model_name -> prediction probabilities
    """
    X_train_scaled, X_test_scaled, scaler = apply_scaler(X_train, X_test, config)
    models = instantiate_models(config)

    class_threshold = config.get("class_threshold", 0.3)
    y_train_bin = binarize_target(y_train, class_threshold)
    y_test_bin = binarize_target(y_test, class_threshold)

    y_pred_dict = train_and_predict(X_train_scaled, y_train_bin, X_test_scaled, models)

    if config.get("save_models", False):
        save_trained_models(models, scaler, config, corpus_test, fold_idx)

    return {
        "sample_id_test": sample_id_test,
        "y_test": y_test,
        "y_test_bin": y_test_bin,
        "y_pred_dict": y_pred_dict
    }

def save_trained_models(models: Dict[str, object],
                        scaler: Optional[object],
                        config: DictConfig,
                        corpus_test: Optional[ndarray],
                        fold_idx: int):
    """
    Save trained models and scaler to disk. Called after each iterative training (e.g., kth fold, kth corpus).

    :param models: Dictionary of model name to trained model.
    :param scaler: Trained scaler object.
    :param config: DictConfig object.
    :param corpus_test: Corpus identifiers for test samples.
    :param fold_idx: Index of the current fold (for k-fold).
    """
    Path(config.retrained_models_path).mkdir(parents=True, exist_ok=True)

    if config.evaluation_type == "leave_one_corpus_out":
        model_subdir = corpus_test[0]
    elif config.evaluation_type == "k_fold_cross_validation":
        model_subdir = f"fold_{fold_idx+1}"
    else:
        model_subdir = "test_set"
    
    (Path(config.retrained_models_path) / model_subdir).mkdir(parents=True, exist_ok=True)

    for model_name, model in models.items():
        # save catboost models with .cbm extension (joblib for others)
        if model_name.lower() == "catboostclassifier":
            model_path = Path(config.retrained_models_path) / model_subdir / f"{model_name}.cbm"
            model.save_model(model_path)
            logger.info("Saved CatBoost model %s to %s", model_name, model_path)
            continue
        
        model_path = Path(config.retrained_models_path) / model_subdir / f"{model_name}.joblib"
        joblib.dump(model, model_path)
        logger.info("Saved model %s to %s", model_name, model_path)
    
    if scaler:
        scaler_path = Path(config.retrained_models_path) / model_subdir / "scaler.joblib"
        joblib.dump(scaler, scaler_path)
        logger.info("Saved scaler to %s", scaler_path)
    
def train_full_dataset(X: ndarray,
                       y: ndarray,   
                       config: DictConfig):
    """
    Train models on the full dataset and save them.

    :param X: Features.
    :param y: Target values.
    :param config: DictConfig object.
    """
    class_threshold = config.get("class_threshold", 0.3)
    y_bin = binarize_target(y, class_threshold)
    X_scaled_full, _, scaler_full = apply_scaler(X, X, config)
    models_full = instantiate_models(config)

    Path(config.retrained_models_path).mkdir(parents=True, exist_ok=True)
    model_subdir = "full_dataset"
    (Path(config.retrained_models_path) / model_subdir).mkdir(parents=True, exist_ok=True)

    for model_name, model in models_full.items():
        try:
            model.fit(X_scaled_full, y_bin)
            if model_name.lower() == "catboostclassifier":
                model_path = Path(config.retrained_models_path) / model_subdir / f"{model_name}.cbm"
                model.save_model(model_path)
                logger.info("Saved full CatBoost model %s to %s", model_name, model_path)
                continue
            
            model_path = Path(config.retrained_models_path) / model_subdir / f"{model_name}.joblib"
            joblib.dump(model, model_path)
            logger.info("Saved full model %s to %s", model_name, model_path)

        except Exception as e:
            logger.error("Error training/saving full model %s: %s", model_name, e)
            raise

    if scaler_full:
        scaler_path = Path(config.retrained_models_path) / model_subdir / "scaler.joblib"
        joblib.dump(scaler_full, scaler_path)
        logger.info("Saved full scaler to %s", scaler_path)
    
def run(config: DictConfig):
    """
    Main training function.

    :param config: DictConfig object.
    """
    X, y, corpus, sample_id, y_bin = prepare_data(config)
    output_data = {}

    for  idx, (train_idx, test_idx) in enumerate(data_split.run(X, y_bin, corpus, config)):
        X_train, y_train = X[train_idx], y[train_idx]
        X_test, y_test = X[test_idx], y[test_idx]
        corpus_test = corpus[test_idx] if corpus is not None else None
        sample_id_test = sample_id[test_idx] if sample_id is not None else None

        result = apply_training(X_train, y_train, X_test, y_test, corpus_test, sample_id_test, config, idx)

        if config.get("evaluate", True):
            if config.evaluation_type == "leave_one_corpus_out":
                output_data[corpus_test[0]] = result
            elif config.evaluation_type == "k_fold_cross_validation":
                output_data[f"fold_{idx+1}"] = result
            else:
                output_data["test_set"] = result

    if config.get("evaluate", True):
        Path(config.evaluation_output_path).mkdir(parents=True, exist_ok=True)
        save_metrics.run(output_data, config)

    if config.get("save_models", False):
        train_full_dataset(X, y, config)

    logger.info("Training and evaluation completed.")