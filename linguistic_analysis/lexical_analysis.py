#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: Daniil Kocharov (dan_ya)
"""
import copy
import csv
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import seaborn as sns
from tqdm import tqdm
from scipy.stats import pearsonr
from typing import List, Union
import stanza
from stanza.models.common.doc import Word
from nltk import corpus


NLP = stanza.Pipeline(lang='en', processors='tokenize,mwt,pos,lemma,depparse')

DATA_DIR = Path('..', 'data')

INPUT_FILE_PATH = DATA_DIR.joinpath('samples_all_text.csv')
FIGURE_PATH = DATA_DIR.joinpath('figures')
FIGURE_PATH.mkdir(parents=True, exist_ok=True)

PARTS_OF_SPEECH = ['all', 'NOUN', 'VERB', 'ADJ', 'ADV', 'PRON']
MIN_N_WORDS = 5
LOG_SHIFT = 1.0

ADAPTIVE_SAMPLE_SELECTION = False
PREDICTION_COLUMN = 'y_pred_prob_ensemble'
ADAPTIVE_FRACTION = 0.30

SKIP_ZERO_COUNTS = False

PLOT_FIGURES = True
PLOT_FOR_PRINT = True
SHOW_PLOTS = False

MAIN_THRESHOLD = 0.6
DECISION_THRESHOLDS = (0, MAIN_THRESHOLD)


def main():
    statistics_df = []
    data_to_plot = dict()
    for suffix in DECISION_THRESHOLDS:
        for pos in PARTS_OF_SPEECH:
            print(f'\n{pos}, {suffix}')

            lex_file = INPUT_FILE_PATH.with_suffix('.lex')
            if lex_file.is_file():
                lex_df = pd.read_pickle(lex_file)
            else:
                print('Parsing the text data...')
                lex_df = pd.read_csv(INPUT_FILE_PATH, sep='\t', quoting=csv.QUOTE_NONE)
                tqdm.pandas(desc='asr text parsing...')
                lex_df['asr_words'] = lex_df['text_large'].progress_apply(get_parsed_words)
                tqdm.pandas(desc='reference text parsing...')
                lex_df['ref_words'] = lex_df['ref_text'].progress_apply(get_parsed_words)
                lex_df.to_pickle(lex_file)

            lex_df[PREDICTION_COLUMN] = lex_df[PREDICTION_COLUMN].astype(float)
            lex_df = lex_df[lex_df[PREDICTION_COLUMN] >= suffix]
            if ADAPTIVE_SAMPLE_SELECTION and suffix == 0:
                lex_df = lex_df.sample(frac=ADAPTIVE_FRACTION)

            lex_df = get_lemma_quantities(lex_df, pos=pos)
            data_to_plot[(pos, suffix)] = copy.deepcopy(lex_df)

            stat_item = {'pos': pos, 'threshold': suffix}
            stat_item.update(get_statistics(copy.deepcopy(lex_df)))

            ttr_data = {'ttr_asr': len(lex_df) / lex_df['asr'].sum(),
                        'ttr_man': len(lex_df) / lex_df['ref'].sum(),
                        }
            stat_item.update(ttr_data)
            statistics_df.append(stat_item)
            words_with_ref_0 = lex_df.loc[lex_df['ref'] == 0, 'word'].tolist()
            words_with_asr_0 = lex_df.loc[lex_df['asr'] == 0, 'word'].tolist()
            print(f' - no ASR: {words_with_asr_0}')
            print(f' - no MANUAL: {words_with_ref_0}')

            mean_val = lex_df['ref'].mean()
            median_val = lex_df['ref'].median()
            print(f' - mean counts of ref for {pos}: {mean_val:.2f}')
            print(f' - median counts of ref for {pos}: {median_val:.2f}')

    statistics_df = pd.DataFrame(statistics_df)
    statistics_df.to_csv(FIGURE_PATH.joinpath('pos_statistics.csv'), sep='\t', index=False)

    if PLOT_FIGURES:
        data_labels = [('NOUN', MAIN_THRESHOLD), ('VERB', MAIN_THRESHOLD), ('NOUN', 0), ('VERB', 0)]
        data_subplots = [data_to_plot[label] for label in data_labels if label in data_to_plot]
        if data_subplots:
            draw_correlation_subplots(data_subplots, FIGURE_PATH.joinpath(f'NOUN_VERB.png'), data_labels)

        for pos, suffix in data_to_plot:
            if pos != 'all':
                continue
            print('Plotting: ', pos, suffix)
            lex_df = data_to_plot[(pos, suffix)]
            fig_file = FIGURE_PATH.joinpath(f'{pos}_{suffix}.png')
            draw_correlation(copy.deepcopy(lex_df), fig_file, pos=pos)

    return


def get_statistics(lex_df: pd.DataFrame) -> dict:
    corr_coef, _ = pearsonr(lex_df['ref'], lex_df['asr'])

    corr_coef_log, _ = pearsonr(np.log10(lex_df['ref'] + LOG_SHIFT), np.log10(lex_df['asr'] + LOG_SHIFT))

    high_freq_df = lex_df[lex_df['asr'] >= MIN_N_WORDS]
    corr_coef_log_high, _ = pearsonr(np.log10(high_freq_df['ref'] + LOG_SHIFT), np.log10(high_freq_df['asr'] + LOG_SHIFT))

    result = {'r': round(corr_coef, 2),
              'r_log': round(corr_coef_log, 2),
              f'r_log_{MIN_N_WORDS}': round(corr_coef_log_high, 2),
              }
    return result


def draw_correlation(lex_df: pd.DataFrame, fig_file: Path, pos: Union[str, None] = None):
    lex_df['ref'] = np.log10(lex_df['ref'] + LOG_SHIFT)
    lex_df['asr'] = np.log10(lex_df['asr'] + LOG_SHIFT)

    high_freq_df = lex_df[lex_df['asr'] >= np.log10(MIN_N_WORDS + LOG_SHIFT)]

    corr_coef, _ = pearsonr(lex_df['ref'], lex_df['asr'])
    corr_coef_high_freq, _ = pearsonr(high_freq_df['ref'], high_freq_df['asr'])

    plt.figure(figsize=(8, 6))
    sns.regplot(data=lex_df, x='ref', y='asr', scatter_kws={'s': 2}, line_kws={'color': 'red'})
    sns.regplot(data=high_freq_df, x='ref', y='asr', scatter_kws={'s': 2}, line_kws={'color': 'blue'})

    # Compute index of the shortest word in each (ref, asr) group
    # Select rows corresponding to those shortest words
    # Annotate only shortest words
    grid_size = 0.2
    lex_df['grid_x'] = (lex_df['ref'] / grid_size * 2 ).round()
    lex_df['grid_y'] = (lex_df['asr'] / grid_size).round()
    lex_df['word_len'] = lex_df['word'].str.len()
    idx = lex_df.groupby(['grid_x', 'grid_y'])['word_len'].idxmax()
    deduped = lex_df.loc[idx]
    for _, row in deduped.iterrows():
        jitter_y = row['asr'] + np.random.uniform(-0.04, 0.04)
        if row['asr'] > np.log10(MIN_N_WORDS):
            plt.text(row['ref'], jitter_y, row['word'], fontsize=10, ha='center', va='bottom')
        else:
            plt.text(row['ref'], jitter_y, row['word'], fontsize=7, ha='center', va='bottom')

    ylim = lex_df['asr'].max()
    xlim = lex_df['ref'].max()
    if pos == 'all':
        ylim = xlim = 3.4
    elif pos == 'NOUN':
        ylim = xlim = 2.2
    elif pos == 'VERB':
        ylim = xlim = 3.0

    # Annotate the correlation coefficient
    plt.text(
        x=-0.1,
        y=ylim * 0.99,
        s=f'r = {corr_coef:.2f}',
        fontsize=12,
        color='red',
        fontweight='bold',
        ha='left',
        va='top'
    )
    plt.text(
        x=-0.1,
        y=ylim * 0.95,
        s=f'high-freq r = {corr_coef_high_freq:.2f}',
        fontsize=12,
        color='blue',
        fontweight='bold',
        ha='left',
        va='top'
    )
    plt.grid(True, which='both', linestyle='--', linewidth=0.5)

    plt.xlabel('Manual transcription word counts (log scale)', fontsize=12, fontweight='bold')
    plt.ylabel('Automatic transcription word counts (log scale)', fontsize=12, fontweight='bold')
    title = f'Pearson Correlation between Reference and ASR ({pos})'
    if not PLOT_FOR_PRINT:
        plt.title(title, fontsize=14)

    if PLOT_FOR_PRINT:
        plt.ylim(-0.1, ylim)
        plt.xlim(-0.25, xlim)

    if PLOT_FOR_PRINT:
        plt.tight_layout()

    if SHOW_PLOTS:
        plt.show()
    if SKIP_ZERO_COUNTS:
        fig_file = fig_file.with_stem(f'{fig_file.stem}_no0')
    plt.savefig(fig_file)
    if PLOT_FOR_PRINT:
        plt.savefig(fig_file.with_suffix('.pdf'))


def draw_correlation_subplots(lex_dfs: list[pd.DataFrame], fig_file: Path, ds_labels: list[tuple[str, str]] = None):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for i, (lex_df, ax) in enumerate(zip(lex_dfs, axes)):
        pos = ds_labels[i][0]
        lex_df['ref'] = np.log10(lex_df['ref'] + LOG_SHIFT)
        lex_df['asr'] = np.log10(lex_df['asr'] + LOG_SHIFT)

        high_freq_df = lex_df[lex_df['asr'] >= np.log10(MIN_N_WORDS + LOG_SHIFT)]

        corr_coef, _ = pearsonr(lex_df['ref'], lex_df['asr'])
        corr_coef_high_freq, _ = pearsonr(high_freq_df['ref'], high_freq_df['asr'])

        sns.regplot(data=lex_df, x='ref', y='asr', ax=ax, scatter_kws={'s': 2}, line_kws={'color': 'red'})
        sns.regplot(data=high_freq_df, x='ref', y='asr', ax=ax, scatter_kws={'s': 2}, line_kws={'color': 'blue'})

        grid_size = 0.3
        lex_df['grid_x'] = (lex_df['ref'] / grid_size * 2).round()
        lex_df['grid_y'] = (lex_df['asr'] / grid_size).round()
        lex_df['word_len'] = lex_df['word'].str.len()
        idx = lex_df.groupby(['grid_x', 'grid_y'])['word_len'].idxmax()
        deduped = lex_df.loc[idx]

        ylim = lex_df['asr'].max()
        xlim = lex_df['ref'].max()
        if pos == 'all':
            ylim = xlim = 3.5
        elif pos == 'NOUN':
            ylim = xlim = 2.2
        elif pos == 'VERB':
            ylim = xlim = 3.0

        for _, row in deduped.iterrows():
            jitter_y = row['asr'] + np.random.uniform(-0.04, 0.04)
            fontsize = 14 if row['asr'] > np.log10(MIN_N_WORDS) else 12
            ax.text(row['ref'], jitter_y, row['word'], fontsize=fontsize, ha='center', va='bottom')

        ax.text(
            x= -0.1,
            y=ylim * 1.05,
            s=f'r = {corr_coef:.2f}',
            fontsize=16,
            color='red',
            fontweight='bold',
            ha='left',
            va='top'
        )
        ax.text(
            x=-0.1,
            y=ylim * 0.98,
            s=f'high-freq r = {corr_coef_high_freq:.2f}',
            fontsize=16,
            color='blue',
            fontweight='bold',
            ha='left',
            va='top'
        )
        selection = 'all' if ds_labels[i][1] == 0 else 'selected'
        title_s = f'{pos.lower()}s ({selection})'
        ax.text(
            x=xlim * 0.45,
            y=ylim * 1.1,
            s=title_s,
            #transform=ax.transAxes,
            fontsize=18,
            color='black',
            fontweight='bold',
            ha='center',
            va='top'
        )

        ax.grid(True, which='both', linestyle='--', linewidth=0.5)
        fig.supxlabel('Manual transcription word counts (log scale)', fontsize=18, fontweight='bold')
        fig.supylabel('ASR transcription word counts (log scale)', fontsize=18, fontweight='bold')

        if PLOT_FOR_PRINT:
            ax.set_ylim(-0.1, ylim * 1.1)
            ax.set_xlim(-0.25, xlim)

    # plt.subplots_adjust(wspace=0.2, hspace=0.2, left=0.07, right=0.97, top=0.95, bottom=0.07)
    plt.subplots_adjust(wspace=0.1, hspace=0.1)
    plt.tight_layout()

    if SHOW_PLOTS:
        plt.show()

    fig_file = fig_file.with_stem(f'{fig_file.stem}_grid') if SKIP_ZERO_COUNTS else fig_file
    fig.savefig(fig_file)
    if PLOT_FOR_PRINT:
        fig.savefig(fig_file.with_suffix('.pdf'))


def get_lemma_quantities(df: pd.DataFrame, pos: str = 'all') -> pd.DataFrame:
    lexemes = {'ref': sum(df['ref_words'].tolist(), []), 'asr': sum(df['asr_words'].tolist(), [])}

    lex_mapping = dict()
    if pos == 'all':
        ref_w = [t for t in lexemes['ref']]
        asr_w = [t for t in lexemes['asr']]
    else:
        ref_w = [t for t in lexemes['ref'] if t.pos == pos]
        asr_w = [t for t in lexemes['asr'] if t.pos == pos]

    for w in ref_w:
        word = w.lemma.lower()
        if word not in lex_mapping:
            lex_mapping[word] = {'ref': 0, 'asr': 0}
        lex_mapping[word]['ref'] += 1
    for w in asr_w:
        word = w.lemma.lower()
        if word not in lex_mapping:
            lex_mapping[word] = {'ref': 0, 'asr': 0}
        lex_mapping[word]['asr'] += 1

    vocabulary = set(corpus.words.words())
    oov_words = [w for w in lex_mapping if w not in vocabulary]
    print(f'OOV words: {oov_words}')

    lex_df = []
    for w in lex_mapping:
        if w not in vocabulary:
            continue
        row = {'word': w, 'ref': lex_mapping[w]['ref'], 'asr': lex_mapping[w]['asr']}
        lex_df.append(row)
    lex_df = pd.DataFrame(lex_df)
    lex_df.sort_values(by='word', inplace=True)
    return lex_df


def get_parsed_words(text: str) -> List[Word]:
    if not type(text) is str:
        return []
    parsed_text = NLP(text)
    words = [w for sentence in parsed_text.sentences for w in sentence.words]
    words = [w for w in words if w.pos != 'PUNCT']
    return words


if __name__ == '__main__':
    main()