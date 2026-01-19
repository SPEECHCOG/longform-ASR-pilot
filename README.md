# Code for D. Kocharov, A. Galama, O. Räsänen (2026). "Enabling automatic transcription of child-centered audio recordings from real-world environments".

Program code for:

 - Training the classifier to detect low-WER transcriptions of spoken utterances.
 - Automatic transcription of spoken utterances
 - Detection of low-WER transcriptions using the trained classifier.
 - Linguistic analysis and comparison of automatic and manual transcriptions.

### 1. Contents

1.1. Automatic transcription
 - `GILES_main.py`: main script to transcribe spoken utterances.
 - `conf`: 
    - `config.yaml`: select the step to perform.
    - `steps`: step-specific parameters.
    - `paths/paths.yaml`: set the paths to the data, models, output results.
    - `features/features.yaml`: list the features for the estimation of transcription quality.
    - `models/models.yaml`: model-specific parameters.

1.2. Linguistic analysis
 - `linguistic_analysis/lexical_analysis.py`: main script to transcribe spoken utterances.

1.3. Sample files to test the transcriber (several files from LJSpeech corpus)
 - `sample/input/eval_corpus`: data to test the classifier trainining and prediction (require audio files along with their transcriptions).
 - `sample/input/transcribe_corpus`: data to test the transcription procedure (require audio files only).

1.4. Lexical data that has been analyzed, including the counts of word lemmas in automatic and corresponding manual transcriptions. All proper names were automatically (by means of Stanza toolkit) and manually removed from the list.
 - `experimental_data/word_all_0.csv`: all automatically transcribed utterances.
 - `experimental_data/word_all_0.6.csv`: selected automatically transcribed utterances.

### 2. Main dependencies

2.1. Classifier
 - whisper-timestamped
 - whisperx

 - librosa
 - hydra-core
 - RapidFuzz

 - catboost
 - lightgbm
 - xgboost

 - pandas
 - scipy
 - numpy

2.2. Linguistic analysis
 - nltk
 - stanza

 - matplotlib
 - seaborn

### 3. Instructions:

3.1. Directory structure

3.1.1. Classifier training and evaluation of previously created transcriptions

The code will run under certain assumptions of the folder structure. If that structure is not follow errors may arise.
The main data folder with the wav files and manual annotations is expected to have folders (corpus) which each contain 
two folders `annotation` and `audio` (see `sample/input/eval_corpus`). In the `audio` folder it is expected to have `.tar` files which each contains the 
recordings in `.wav` format and in the `annotation` folder is expected to have `.csv` files with the manual annotations 
for each `.tar` file in `audio`. They should have the same name, it can be any name, but has to be the same.

There can be several `.tar` files as they could indicate sessions (one corpus made of several sessions). Similarly, there
could be several corpora, but each should have the same folder structure. Corpora can have any name as long as they are 
different. See examples below.

It is mandatory that the annotations files are csv files using `\t` as separator. The files must have `file` and 
`ref_text` (transcription). However, other meta-information is possible to have,
but check that the column names are not in the list of features, see `conf/features/features.yaml`: 

```bash
.
├── corpusA
│   ├── annotation
│   │   ├── 345-ab.csv
│   │   ├── 862-cd.csv
│   │   ├── a21-23.csv
│   │   └── ab-rt2.csv
│   └── audio
│       ├── 345-ab.tar
│       ├── 862-cd.tar
│       ├── a21-23.tar
│       └── ab-rt2.tar
├── corpusB
│   ├── annotation
│   │   └── files1.csv
│   ├── audio
│   │   └── files1.tar

```

3.1.2. Transcription and evaluation of the transcriptions

The corpus directory should contain `.wav` files to be processed. See examples below.

```bash
.
├── corpusA
│       ├── file1.wav
│       ├── file2.wav
│       ├── file3.wav
│       └── file4.wav
├── corpusB
│   ├── part1
│   │   ├── file1.wav
│   │   ├── file2.wav
│   │   ├── file3.wav
│   │   └── file4.wav
│   ├── part2
│   │   ├── file5.csv
│   │   ├── file6.csv
│   │   ├── file7.csv
│   │   └── file8.csv
```


3.2. Executing the code

The code has been developed using [hydra](https://github.com/facebookresearch/hydra) therefore, the parameters for the execution are organise in 
configuration files. However, note that parameters can be given in the command line using hydra override feature. 

The code is located in `src`. The main script is `run.py` and the configuration files are located in 
`conf`.
You can run as follows:

```bash
python run.py
```

if you want to run `transcribe` you have two options, 1) you update the `conf/config.yaml` file and specify the step in 
the `steps` param or 2) you override it in the command line.

**Option 1**
```yaml
defaults:
  - steps:
      - transcribe
  - _self_
```

**Option 2**
```bash
python run.py 'steps=[transcribe]'
```
