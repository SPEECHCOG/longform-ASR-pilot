#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    @author María Andrea Cruz Blandón (andrea08)
"""
import hydra
import logging

from omegaconf import DictConfig
from pathlib import Path
from src.feature_extraction.wer import calculate_wer
from src.feature_extraction.asr import asr
from src.feature_extraction.utils import join_data_together

logger = logging.getLogger(__name__)

def feature_extraction(params: DictConfig, transcribe_mode: bool=False):
    logger.info("Initialising feature extraction")
    data_path = Path(params.input_path).resolve()
    if not data_path.exists():
        logger.error(f"Data path does not exist: {data_path}")
        exit(1)

    if transcribe_mode:
        corpus_dirs = [data_path]
        params.clear_tmp = False
    elif params.corpus_paths:
        corpus_dirs = [data_path.joinpath(d) for d in params.corpus_paths]
    else:
        corpus_dirs = [d for d in data_path.iterdir() if d.is_dir()]
        
    valid_dirs = []
    if transcribe_mode:
        valid_dirs = [d.resolve() for d in corpus_dirs]
    else:
        for corpus_dir in corpus_dirs:
            if not corpus_dir.exists():
                logger.warning(f"{corpus_dir} does not exist, skipping")
                continue
            if corpus_dir.is_dir():
                if not corpus_dir.joinpath('audio').is_dir():
                    logger.warning(f"{corpus_dir}: no audio folder, skipping")
                    continue
                if not corpus_dir.joinpath('annotation').is_dir():
                    logger.warning(f"{corpus_dir}: no annotation folder, skipping")
                    continue
            valid_dirs.append(corpus_dir.resolve())
    if len(valid_dirs) == 0:
        logger.error("No corpus directories found, exiting")
        exit(1)

    output_dir = Path(params.output_path)
    if "whisper" in params.features:
        params_feats = {}
        if "overwrite" in params.keys():
            params_feats["overwrite"] = params.overwrite
        if "clear_tmp" in params.keys():
            params_feats["clear"] = params.clear_tmp
        if "alignment_confidence" in params.whisper.keys():
            params_feats["alignment_confidence"] = params.whisper.alignment_confidence
        if "language" in params.whisper.keys():
            params_feats["language"] = params.whisper.language
        if "small" in params.whisper.keys():
            logger.info("Extracting Whisper small features")
            asr.run(valid_dirs, output_dir, params.whisper.small, **params_feats)
        if "large" in params.whisper.keys():
            logger.info("Extracting Whisper large features")
            asr.run(valid_dirs, output_dir, params.whisper.large, **params_feats)
    if "wer" in params.features:
        params_feats = {}
        if "overwrite" in params.keys():
            params_feats["overwrite"] = params.overwrite
        if "target" in params.wer.keys():
            params_feats["target"] = params.wer.target
        if "dist_reference" in params.wer.keys():
            params_feats["dist_reference"] = params.wer.dist_reference
        calculate_wer.run(valid_dirs, output_dir, **params_feats)
    if "joint_report" in params.features:
        params_feats = {}
        if "overwrite" in params.keys():
            params_feats["overwrite"] = params.overwrite
        if "drop_na_wer" in params.joint_report.keys():
            params_feats["drop_na_wer"] = params.joint_report.drop_na_wer
        if "include_annotations" in params.joint_report.keys():
            params_feats["include_annotations"] = params.joint_report.include_annotations
        if transcribe_mode:
            params.joint_report.features = [f for f in params.joint_report.features if f not in ['wer']]
            params_feats["include_annotations"] = False
        join_data_together.run(data_path, output_dir, params.joint_report.features, **params_feats)
    else:
        logging.warning(f'The feature specify is not implemented yet. Nothing will be done. Exiting')
        exit(1)


def run(config: DictConfig, transcribe_mode: bool=False):
    feature_extraction(config, transcribe_mode)