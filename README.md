# Video Summarisation & Quiz Generation for Educational Content

An end-to-end NLP pipeline that transforms lecture videos into concise summaries and auto-generates multiple-choice quizzes — making revision faster and smarter.

## Pipeline Overview

**Stage 1 — Input & Extraction**
```
Lecture Video (YouTube / Kaggle)  →  Video Download  →  Audio Extraction (FFmpeg)
```

**Stage 2 — Transcription & Text Processing**
```
Audio  →  Speech-to-Text (Whisper)  →  Raw Transcript
       →  Text Cleaning  →  Sentence Segmentation  →  Redundancy Removal
```

**Stage 3 — Summarisation**
```
Filtered Sentences
    →  Chunking with Overlap (20-sentence windows)
    →  Chunk-wise Summarisation (BART)
    →  Combine Chunk Summaries
    →  Final Global Summarisation (BART)
    →  Post-processing
    →  Final Summary Output
```

**Stage 4 — Question Generation**
```
Summary Input
    →  Answer Extraction (POS tagging, noun phrases)
    →  Question Generation (T5)
    →  Distractor Generation (noun phrases from other summary sentences)
    →  Duplicate Question Removal (embedding similarity)
    →  Final MCQ Output
```

**Evaluation** *(runs in parallel with summarisation)*
```
Summary + Original Transcript  →  ROUGE Score Calculation  →  Performance Metrics Output
```

## Pretrained Models Used

| Model | Role |
|---|---|
| `faster-whisper` (Whisper `base`) | Speech-to-text transcription |
| `all-MiniLM-L6-v2` (SentenceTransformers) | Sentence embeddings for deduplication |
| `facebook/bart-large-cnn` | Chunked abstractive summarisation |
| `mrm8488/t5-base-finetuned-question-generation-ap` | Question generation from summary sentences |
| `lmqg/t5-base-squad-qg` | Context-aware question generation |

## Features

- **Flexible input**: accepts YouTube URLs or Kaggle lecture video datasets
- **Filler word removal**: cleans `uh`, `um`, `you know`, etc. from transcripts
- **Semantic deduplication**: removes near-duplicate sentences using cosine similarity (threshold 0.8)
- **Chunked summarisation**: splits long transcripts into overlapping 20-sentence windows before passing to BART, ensuring full coverage
- **MCQ generation**: extracts noun-phrase answers via POS tagging; distractors sourced from other summary sentences (no hardcoded word lists)
- **ROUGE evaluation**: scores each summary against the original transcript

## Setup

```bash
pip install yt-dlp faster-whisper sentence-transformers nltk rouge-score transformers sentencepiece gensim kagglehub
```

FFmpeg must also be installed and available on your PATH.

## Usage

Open `nlp_project_polished.ipynb` and run cells in order.

- Set `SOURCE = 'youtube'` and provide a YouTube URL in Cell 3A, **or**
- Set `SOURCE = 'kaggle'` to use the Kaggle lecture video dataset (Cell 3B)

Output folders created automatically: `transcripts/`, `summaries/`, `quiz_output/`

## Evaluation

Summaries are evaluated with ROUGE-1, ROUGE-2, and ROUGE-L against the original transcript.

## Tech Stack

Python · Jupyter · HuggingFace Transformers · SentenceTransformers · faster-whisper · NLTK · ROUGE
