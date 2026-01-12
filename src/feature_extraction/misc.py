#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
    @author María Andrea Cruz Blandón (andrea08)
"""
whisper_large_ext = '.asr_large'
whisper_small_ext = '.asr_small'
wer_dist_ext = '.wer_dist'
wer_ext = '.wer'

data_sources = {
    'wer_dist': {'label': 'comparison of ASR by small and large models', 'ext': wer_dist_ext, 'sep': '\t', 'drop': None},
    'wer': {'label': 'WER from manual annotations and ASR model', 'ext': wer_ext, 'sep': '\t', 'drop': None},
    }
