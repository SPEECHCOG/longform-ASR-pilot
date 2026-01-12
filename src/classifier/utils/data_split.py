"""
sr/classifier/utils/data_split.py

Handles splitting data for evaluation:
- leave-one-corpus-out
- stratified k-fold cross-validation
- simple train-test split
"""

import logging
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold

logger = logging.getLogger(__name__)

def leave_one_corpus_out(corpus: np.ndarray):
    """
    Generator for leave-one-corpus-out splits.

    :param corpus: Array of corpus IDs.
    :return: Yields (train_idx, test_idx) tuples.
    """
    if corpus is None:
            logger.error("Corpus IDs required for leave-one-corpus-out splitting")
            raise ValueError()
    
    unique_corpora = np.unique(corpus)

    for c in unique_corpora:
        test_idx = np.where(corpus == c)[0]
        all_indices = np.arange(len(corpus))
        train_idx = np.setdiff1d(all_indices, test_idx)
        logger.info("Leave-one-corpus-out split: test corpus=%s, train=%d, test=%d", c, len(train_idx), len(test_idx))
        yield train_idx, test_idx


def stratified_k_fold(X: np.ndarray, y: np.ndarray, k: int):
    """
    Generator for stratified k-fold cross-validation splits.
    If target appears continuous, falls back to standard K-Fold.
    
    :param X: Feature matrix.
    :param y: Target array.
    :param k: Number of folds.
    :return: Yields (train_idx, test_idx) tuples.
    """
    try:
        kf = StratifiedKFold(n_splits=k, shuffle=True, random_state=42)
        _ = list(kf.split(X, y)) # or _ = next(kf.split(X, y))
    except ValueError:
        logger.warning("Stratified K-Fold failed; falling back to standard K-Fold.")
        kf = KFold(n_splits=k, shuffle=True, random_state=42)

    for fold, (train_idx, test_idx) in enumerate(kf.split(X, y)):
        logger.info("K-Fold split %d: train=%d, test=%d", fold + 1, len(train_idx), len(test_idx))
        yield train_idx, test_idx

def simple_split(X: np.ndarray, y: np.ndarray, test_size: float):
    """
    Single train-test split with stratification fallback.

    Falls back to standard split if:
      - Target appears continuous, or
      - Stratified split fails (e.g. too few samples per class).

    :param X: Feature matrix.
    :param y: Target array.
    :param test_size: Proportion of data to use as test set.
    :return: Yields (train_idx, test_idx) tuple.
    """
    try:
        train_idx, test_idx = train_test_split(
            np.arange(len(X)),
            test_size=test_size,
            stratify=y,
            random_state=42,
        )
    except ValueError as e:
        logger.warning(
            "Stratified split failed (%s). Falling back to standard split.", e
        )
        train_idx, test_idx = train_test_split(
            np.arange(len(X)),
            test_size=test_size,
            random_state=42,
        )

    logger.info("Train-test split: train=%d, test=%d", len(train_idx), len(test_idx))
    yield train_idx, test_idx

def run(X, y, corpus, config):
    """
    Run the data splitting according to config.
    Returns an iterator over (train_idx, test_idx).
    Raises ValueError for invalid configurations.
    """
    eval_type = str(config.evaluation_type)

    if y is None:
        logger.error("Target labels are required for data splitting")
        raise ValueError()


    if eval_type == "leave_one_corpus_out":
        logger.info("Using leave-one-corpus-out evaluation ...")
        return leave_one_corpus_out(corpus)

    elif eval_type == "k_fold_cross_validation":
        logger.info("Using %d folds for k-fold cross-validation ...", config.k_folds)
        return stratified_k_fold(X, y, k=config.k_folds)

    elif eval_type == "train_test_split":
        logger.info("Using train-test split with test_size=%.2f ...", config.test_size)
        return simple_split(X, y, test_size=config.test_size)

    else:
        logger.error("Unknown evaluation_type: %s", eval_type)
        raise ValueError()