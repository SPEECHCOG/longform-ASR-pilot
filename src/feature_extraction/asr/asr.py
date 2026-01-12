#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniil Kocharov (dan_ya)
"""
import gc
import shutil
import numpy as np
from librosa import get_duration
import logging
import pandas as pd
from pathlib import Path
import torch
import tarfile
import soundfile as snd
import whisper_timestamped as whisper
import whisperx
from typing import Union, List, Dict, Tuple
from csv import QUOTE_NONE 

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def run(
    audio_dirs: List[Union[Path, str]],
    output_dir: Union[Path, str],
    model_cfg: Dict,
    alignment_confidence: bool = True,
    language: str = 'en',
    overwrite: bool = False,
    clear: bool = True
    ):
    """Runs ASR transcription on the given audio directories."""

    model_asr = None
    model_align = None
    metadata_align = None

    for ds_path in sorted(audio_dirs):
        logging.info(f"Processing dataset: {ds_path.stem}")
        ds_output_dir = output_dir.joinpath(ds_path.stem)
        ds_output_dir.mkdir(parents=True, exist_ok=True)

        audio_paths = list(ds_path.joinpath('audio').glob('*.tar'))
        if not audio_paths:
            audio_paths = [ds_path]
        for audio_path in sorted(audio_paths):
            logging.info(f"Processing audio archive: {audio_path.name}")
            output_file = ds_output_dir.joinpath(f'{audio_path.stem}.asr_{model_cfg["suf"]}.csv')
            tar_output_file = ds_output_dir.joinpath(f'{audio_path.stem}.asr_{model_cfg["suf"]}')

            if not overwrite and output_file.is_file() and tar_output_file.is_file():
                logging.info(f"Skipping {audio_path.stem} (asr: already processed)")
                continue

            if audio_path.is_file() and audio_path.suffix == '.tar':
                tmp_dir = audio_path.with_suffix(f'.tmp_asr_{model_cfg["suf"]}')
                tmp_dir.mkdir(parents=True, exist_ok=True)
                with tarfile.open(audio_path, 'r') as tar:
                    tar.extractall(path=tmp_dir)
            else:
                tmp_dir = audio_path
                clear = False

            if model_asr is None:
                device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
                logging.info(f"Using device: {device}")
                model_asr = whisper.load_model(model_cfg['model'], device=device)
                if alignment_confidence:
                    num_gpus = torch.cuda.device_count()
                    align_device = torch.device(f"cuda:1") if num_gpus > 1 else device
                    model_align, metadata_align = whisperx.load_align_model(language_code=language, device=str(align_device))

            transcripts = []
            audio_files = sorted(tmp_dir.rglob('*.wav'))
            for i, file in enumerate(audio_files):
                logging.info(f"Processing ({i+1}/{len(audio_files)}): {file.stem}")

                audio, _ = snd.read(file)
                result = whisper.transcribe(model_asr, audio, language=language, detect_disfluencies=True,
                    beam_size=5, best_of=5, temperature=(0.0, 0.2, 0.4, 0.6, 0.8, 1.0))

                c_mean, c_min, c_max = get_confidence_values(result, 'segments')
                transcript_entry = {
                    'file': file.stem,
                    'duration': get_duration(path=file),
                    'text': ' '.join(s['text'] for s in result.get('segments', [])),
                    'confidence_mean': c_mean,
                    'confidence_min': c_min,
                    'confidence_max': c_max,
                }
                if alignment_confidence:
                    signal_to_align = whisperx.load_audio(str(file))
                    result['whisperx_segments'] = []
                    try:
                        whisperx_result = whisperx.align(result['segments'], model_align, metadata_align, signal_to_align, align_device, return_char_alignments=False)
                        result['whisperx_segments'] = whisperx_result['segments']
                    except Exception as e:
                        logging.info(f"ERROR: {e}")
                    c_mean, c_min, c_max = get_confidence_values(whisperx_result, 'segments', 'score')
                    transcript_entry['confidence_align_mean'] = c_mean
                    transcript_entry['confidence_align_min'] = c_min
                    transcript_entry['confidence_align_max'] = c_max
                transcripts.append(transcript_entry)

            df_columns_to_save = ['file', 'duration', 'text', 'confidence_mean', 'confidence_min', 'confidence_max']
            if alignment_confidence:
                df_columns_to_save += ['confidence_align_mean', 'confidence_align_min', 'confidence_align_max']

            df = pd.DataFrame(transcripts)
            df.to_csv(output_file, columns=df_columns_to_save, sep='\t', index=False, quoting=QUOTE_NONE)

            if clear:
                shutil.rmtree(tmp_dir)
    if model_asr is not None:
        del model_asr
    if model_align is not None:
        del model_align
    gc.collect()
    torch.cuda.empty_cache()
    return None


def get_confidence_values(data: dict, segment_label: str = 'segments', c_label: str = 'confidence') -> Tuple[float, float, float]:
    c_values = [w[c_label] for s in data[segment_label] for w in s.get('words', []) if c_label in w]
    c_mean = float(np.mean(c_values)) if c_values else 0.0
    c_min = float(min(c_values)) if c_values else 0.0
    c_max = float(max(c_values)) if c_values else 0.0
    return c_mean, c_min, c_max



