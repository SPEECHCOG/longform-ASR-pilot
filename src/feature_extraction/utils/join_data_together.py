#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniil Kocharov (dan_ya)
"""
import logging
import pandas as pd
from pathlib import Path
from typing import Union, List, Dict
from csv import QUOTE_NONE
import math
from ..misc import data_sources

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')


def run(
    annotation_dir: Union[Path, str], 
    output_dir: Union[Path, str], 
    data_types: List[str],
    drop_na_wer: bool = True,
    include_annotations: bool = True, 
    overwrite: bool = False
):
    """
    Process and merge feature data from multiple sources, optionally including annotation data.
    """
    all_data = []
    data_stat = []
    output_dirs = [d for d in Path(output_dir).iterdir() if d.is_dir()]
    all_ext = '.all'
    first_ext = data_sources[data_types.pop(0)]['ext']

    for ds_path in sorted(output_dirs):
        ds_path = ds_path.resolve()
        logging.info(f'Processing dataset: {ds_path.stem}')
        feat_files = list(ds_path.glob(f"*{first_ext}.csv"))

        for feat_file in sorted(feat_files):
            logging.info(f"Processing file (join data): {feat_file.stem.split('.')[0]}")
            output_file = feat_file.with_stem(feat_file.stem.replace(first_ext, all_ext))

            if not overwrite and output_file.is_file():
                file_data = pd.read_csv(output_file, sep='\t', quoting=QUOTE_NONE).fillna('')
            else:
                file_data = pd.read_csv(feat_file, sep='\t', quoting=QUOTE_NONE).fillna('')

                for d in data_types:
                    file_data = merge_data(file_data, feat_file, data_sources[d], first_ext)

                annotation_file = annotation_dir.joinpath(ds_path.stem, 'annotation', f'{feat_file.stem.replace(first_ext, "")}.csv')
                if include_annotations:
                    # todo validation
                    annotation_data = pd.read_csv(annotation_file, sep='\t', quoting=QUOTE_NONE).fillna('')
                    annotation_data.drop(columns=['ref_text'], errors='ignore', inplace=True)
                    file_data = pd.merge(file_data, annotation_data, how='left', on='file')
                else:
                    logging.warning(f'Annotation file not found: {annotation_file}')
                logging.info(f'Data size: {len(file_data)}')

                if 'duration' not in file_data.columns and 'duration_large' in file_data.columns:
                    file_data['duration'] = file_data['duration_large']
                file_data.drop(columns=['duration_large', 'duration_small'], errors='ignore', inplace=True)

                if 'wer' in file_data.columns:
                    data_stat.append({'file': feat_file.stem, 'all': len(file_data), 'wer': file_data['wer'].ne('').sum()})
                elif 'wer_model_dist' in file_data.columns:
                    data_stat.append({'file': feat_file.stem, 'all': len(file_data), 'wer': file_data['wer_model_dist'].ne('').sum()})

                if drop_na_wer and 'wer' in file_data.columns:
                    file_data = file_data[file_data['wer'].ne('')]

                # Reorder columns
                text_cols = [col for col in file_data if 'text' in col]
                if 'ref_text' in text_cols:
                    text_cols.insert(0, text_cols.pop(text_cols.index('ref_text')))
                text_cols = ['file'] + text_cols
                file_data = file_data[text_cols + [col for col in file_data.columns if col not in text_cols]]

                file_data.to_csv(output_file, sep='\t', index=False, quoting=QUOTE_NONE)
            all_data.append(file_data)
    if data_stat:
        pd.DataFrame(data_stat).to_csv(output_dir.joinpath('data_stat.csv'), sep='\t', index=False, quoting=QUOTE_NONE)
    df = pd.concat(all_data, ignore_index=True).fillna('')
    df.to_csv(output_dir.joinpath(f'_{output_dir.stem}_data.csv'), sep='\t', index=False, quoting=QUOTE_NONE)
    return


def merge_data(data: pd.DataFrame, file: Path, setup: Dict[str, Union[str, List[str], None]], file_ext:str) -> pd.DataFrame:
    data_file = file.with_stem(file.stem.replace(file_ext, setup['ext']))

    assert data_file.is_file(), f'No associate {setup["label"]} for file: {file} ({data_file})'

    new_data = pd.read_csv(data_file, sep=setup['sep'], quoting=QUOTE_NONE)
    if setup['drop'] is not None:
        new_data.drop(columns=setup['drop'], inplace=True)

    for col in ['text_large', 'text_small']:
        if col in new_data and col in data:
            new_data.drop(columns=[col], inplace=True)

    data = pd.merge(data, new_data, on='file', how='left')
    return data
