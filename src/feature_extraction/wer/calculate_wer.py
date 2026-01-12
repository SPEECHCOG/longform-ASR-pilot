#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniil Kocharov (dan_ya)
"""
import logging
import pandas as pd
import re
from pathlib import Path
from typing import Union, List, Tuple
from rapidfuzz.distance.Levenshtein import normalized_distance, editops
from csv import QUOTE_NONE


out_ext = '.wer_dist'
dist_col_id = 'wer_model_dist'
target_col_id = 'wer'


def run(
    annotation_dirs:List[Union[Path, str]],
    output_dir:Union[Path, str],
    dist_reference:str = 'large',
    unit_to_compare:str = 'word',
    target:str = '',
    overwrite:bool = False,
    ):
    """Runs WER distance calculation between ASR outputs and annotations."""

    first_ext = '.asr_large' if dist_reference == 'large' else '.asr_small'
    second_ext = '.asr_small' if dist_reference == 'large' else '.asr_large'

    output_dirs = [d for d in Path(output_dir).iterdir() if d.is_dir()]
    for ds_path in sorted(output_dirs):
        logging.info(f"Processing dataset (wer): {ds_path.stem}")

        first_files = sorted(ds_path.glob(f'*{first_ext}.csv'))
        for first_file in first_files:
            second_file = first_file.with_stem(first_file.stem.replace(first_ext, second_ext)).with_suffix('.csv')
            output_file = first_file.with_stem(first_file.stem.replace(first_ext, out_ext))

            if not second_file.is_file():
                logging.warning(f"Missing counterpart file: {second_file}")
                continue

            if not overwrite and output_file.is_file():
                logging.info(f"Skipping {first_file.stem} (wer_dist: already processed)")
                continue

            first_data = pd.read_csv(first_file, sep='\t', quoting=QUOTE_NONE).fillna('')
            second_data = pd.read_csv(second_file, sep='\t', quoting=QUOTE_NONE).fillna('')
            file_data = pd.merge(first_data, second_data, on='file', how='left', suffixes=('_large', '_small'))
            file_data[['wer_model_dist', 'ins_model_dist', 'del_model_dist', 'sub_model_dist']] = file_data.apply(
                lambda row: pd.Series(distance_calculation(row['text_large'], row['text_small'], unit=unit_to_compare)) if row['text_large'] and row['text_small']
                    else pd.Series(['1.0', '1.0', '1.0', '1.0']), axis=1
            )
            file_data['n_words'] = file_data['text_large'].apply(
                lambda x: len(x.strip().split()) if isinstance(x, str) and x.strip() else 0
            )

            file_data.to_csv(output_file, sep='\t', index=False, quoting=QUOTE_NONE)

    if target != '':
        target_ext = '.asr_large' if target == 'large' else '.asr_small'
        for ds_path in sorted(annotation_dirs):
            logging.info(f"Processing annotations (wer): {ds_path.stem}")

            annotation_paths = sorted(ds_path.joinpath('annotation').glob('*.csv'))
            for annotation_file in annotation_paths:
                second_file = output_dir.joinpath(ds_path.stem, f'{annotation_file.stem}{target_ext}.csv')
                output_file = output_dir.joinpath(ds_path.stem, f'{annotation_file.stem}.wer.csv')

                if not second_file.is_file():
                    logging.warning(f"Missing ASR file: {second_file}")
                    continue

                if not overwrite and output_file.is_file():
                    logging.info(f"Skipping {annotation_file.stem} (already processed)")
                    continue

                annotations = pd.read_csv(annotation_file, usecols=['file', 'ref_text'], sep='\t', quoting=QUOTE_NONE).fillna('')
                model_data = pd.read_csv(second_file, usecols=['file', 'text'], sep='\t', quoting=QUOTE_NONE).fillna('').rename(columns={'text': f'text_{target}'})

                annotations = annotations.merge(model_data, on='file', how='inner')
                annotations[['wer', 'ins', 'del', 'sub']] = annotations.apply(
                    lambda row: pd.Series(distance_calculation(row['ref_text'], row[f'text_{target}'], unit=unit_to_compare)) if row['ref_text'] and row[f'text_{target}']
                    else pd.Series(['1.0', '1.0', '1.0', '1.0']), axis=1
                )
                annotations.to_csv(output_file, sep='\t', index=False, quoting=QUOTE_NONE)
    return None


def distance_calculation(input_text1, input_text2, unit='word') -> Tuple[float, Union[float, None], Union[float, None], Union[float, None]]:
    text1 = text_norm(input_text1, unit=unit)
    text2 = text_norm(input_text2, unit=unit)
    dist = normalized_distance(text1, text2)
    ops = editops(text1, text2)
    insertions = sum(1 for op in ops if op[0] == 'insert') / len(text1) if text1 else None  # or len(ops)
    deletions = sum(1 for op in ops if op[0] == 'delete') / len(text1) if text1 else None  # or len(ops)
    substitutions = sum(1 for op in ops if op[0] == 'replace') / len(text1) if text1 else None  # or len(ops)
    return dist, insertions, deletions, substitutions


def text_norm(text: str, unit: str = 'word') -> Union[List[str], str]:
    """Splits text into words or character n-grams."""
    result = clear_text(text).lower()
    if unit == '3gram':
        result = [result[i:i+3] for i in range(len(result) - 2)]
    elif unit == '2gram':
        result = [result[i:i+2] for i in range(len(result) - 1)]
    elif unit == '1gram':
        result = [result[i:i+1] for i in range(len(result))]
    else:  # split into words by space
        result = result.split(' ')
    return result


def clear_text(input_text: str) -> str:
    """Cleans transcript text by removing non-linguistic symbols."""

    if not input_text:
        return input_text

    input_text = re.sub(r'\[\s*:\s*', '[:', input_text)
    words = re.split(r'\s+', input_text)
    remove_words = []
    for i in range(len(words)):
        correction = re.match(r'\[:(.+?)\]', words[i])
        if correction:
            words[i-1] = correction[1]
            remove_words.append(i)
        words[i] = re.sub(r'<.+?>', '', words[i])
        words[i] = re.sub(r'@.+', '', words[i])
        words[i] = re.sub(r'xxx', '', words[i])
        words[i] = re.sub(r'&=.+', '', words[i])
        words[i] = re.sub(r'^\W+', '', words[i])
        words[i] = re.sub(r'\W+$', '', words[i])
    if remove_words:
        for i in reversed(remove_words):
            del words[i]
    words = [w for w in words if w != '']
    result = ' '.join(words)
    return result
