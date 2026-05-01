# Bertalign-Fast

An automatic mulitlingual sentence aligner optimized for CPU.

## Overview

Bertalign-Fast is a lightweight, CPU-optimized version of [Bertalign](https://github.com/bfsujason/bertalign) that uses [modern SWE](https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1) (Static Word Embeddings) instead of CWE (Contextualized Word Embeddings).

### Key Features

- 🚀 **Fast on CPU** - No GPU required
- 💡 **Lightweight** - Use static word embeddings
- 🌍 **Multilingual** - Support 30+ languages
- 🎯 **Accurate** - Maintain high alignment quality
- 🖥️ **User-friendly GUI** - Visual interface for alignment tasks
- 🔄 **Multi-version alignment** - Align multiple language pairs or document versions

### When to Use Which?

- Use Bertalign when:
  - You have GPU available
  - Maximum accuracy is critical
  - Processing time is not a constraint
  - Working with complex literary or highly nuanced texts

- Use Bertalign-Fast when:
  - You need fast CPU inference
  - Processing large volumes of text
  - Running on resource-constrained systems
  - Working with non-literary texts (news, technical documents, etc.)

## Installation

```bash
git clone https://github.com/bfsujason/bertalign_fast.git

cd bertalign_fast

# Core only
pip install -r requirements.txt

# If you want the GUI
pip install pyqt5 igraph

# Download SWE model
python download_model.py

# Use mirror site if you cannot visit Hugging Face
python download_model.py --mirror hf-mirror.com
```

## Quick Start

```python
from bertalign_fast import BertalignFast

aligner = BertalignFast()

aligner.align_sents(src_text, tgt_text)

print(aligner.result)
```

