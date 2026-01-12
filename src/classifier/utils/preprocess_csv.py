"""
src/classifier/utils/preprocess_csv.py

Handles data loading and preprocessing for training and prediction steps.
"""
import csv
import logging
import pandas as pd
from pathlib import Path
from omegaconf import DictConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def load_csv(csv_file: str, config: DictConfig) -> pd.DataFrame:
    """
    Load CSV file into a pandas DataFrame with correct dtypes and optional OpenSMILE features.
    """
    df = pd.read_csv(
        csv_file,
        sep="\t", #","
        header=0, 
        skip_blank_lines=False,
        engine="python",
        quoting=csv.QUOTE_NONE,
    )
    if str(config.corpus_id) in df.columns:
        df = df.dropna(subset=[str(config.corpus_id)])
    return df


def extract_features(df: pd.DataFrame, config: DictConfig, mode: str) -> tuple:
    """
    Extracts features from the DataFrame and applies filtering based on config settings.
    
    :param df: Input DataFrame.
    :param config: DictConfig object with feature extraction settings.
    :param mode: Either "training" or "prediction".
    :return: tuple (X, y, corpus, sample_id)
    """
    features = list(config.features)
    target_col = str(config.target)
    corpus_col = str(config.corpus_id)
    sample_id_col = str(config.sample_id)

    # Extract X
    try:
        X = df[features].copy()
    except KeyError as e:
        raise KeyError(f"Missing required feature column in DataFrame: {e}")

    # Drop rows with NaN in features
    X = X.dropna()

    # Apply duration filtering
    if "duration" in df.columns:
        before = len(X)
        X = X[X["duration"] >= config.duration_threshold]
        after = len(X)
        logger.info("Filtered %d rows by duration threshold (kept %d)", before - after, after)

    y = None
    corpus = None
    sample_id = None
    if mode in ('training', 'prediction'):
        # Handle target (WER or class label)
        if target_col and target_col in df.columns:
            y = df.loc[X.index, target_col].copy()
        else:
            if mode == "training":
                logger.error("Target column '%s' not found in DataFrame for training mode.", target_col)
                raise KeyError(f"Target column '{target_col}' not found in DataFrame.")
            else:
                logger.warning("No target column found — proceeding without labels.")
        
        # Handle corpus
        if corpus_col and corpus_col in df.columns:
            corpus = df.loc[X.index, corpus_col].copy()
        else:
            logger.warning("No corpus ID column found — leave-one-corpus-out evaluation will not be possible.")

    # Handle file names
    if sample_id_col and sample_id_col in df.columns:
        sample_id = df.loc[X.index, sample_id_col].copy()
    else:
        logger.warning("No sample ID column found — sample identification may be difficult.")
    return X, y, corpus, sample_id


def run(config: DictConfig, mode: str = "training") -> tuple:
    """
    Main preprocessing entrypoint.
    - Loads the data
    - Extracts features, labels (if available), and corpus IDs
    - Handles training vs prediction vs transcription differences
    
    :param config: DictConfig object.
    :param mode: Either "training" or "prediction".
    :return: tuple (X, y, corpus, sample_id)
    """
    logger.info("Starting preprocessing in %s mode", mode)

    data_path = Path(config.extracted_features_path).resolve()
    if data_path.stem != f'_{data_path.parent.stem}_data':
        data_path = data_path.joinpath(f'_{data_path.stem}_data.csv')
    df = load_csv(data_path, config)
    X, y, corpus, sample_id = extract_features(df, config, mode)
    transcriptions = None
    if mode == 'transcription' and 'text_large' in df.columns:
        transcriptions = df[['file','text_large']]

    logger.info("Preprocessing complete: %d samples, %d features", len(X), X.shape[1])
    return X, y, corpus, sample_id, transcriptions
