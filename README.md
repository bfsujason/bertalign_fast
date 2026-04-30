# Bertalign-Fast

An automatic mulitlingual sentence aligner optimized for CPU.

## Overview

Bertalign-Fast is a lightweight, CPU-optimized version of [Bertalign](https://github.com/bfsujason/bertalign) that uses [modern static word embeddings](https://huggingface.co/sentence-transformers/static-similarity-mrl-multilingual-v1) instead of transformer-based models.

### Key Features

- 🚀 **Blazing fast on CPU** - No GPU required
- 💡 **Lightweight** - Use static word embeddings
- 🎯 **Accurate** - Maintains high alignment quality
- 🖥️ **User-friendly GUI** - Visual interface for alignment tasks
- 🔄 **Multi-version alignment** - Align multiple language pairs or document versions

### When to Use Which?

- Use Bertalign when:
  - You have GPU available
  - Maximum accuracy is critical
  - Processing time is not a constraint

- Use Bertalign-Fast when:
  - You need fast CPU inference
  - Processing large volumes of text
  - Running on resource-constrained systems

## Installation

```bash
git clone https://github.com/bfsujason/bertalign_fast.git
cd bertalign_fast
pip install -r requirements.txt
python download_model.py
```

## Quick Start

```python
from bertalign_fast import BertalignFast

aligner = BertalignFast()

aligner.align_sents(src_text, tgt_text)

print(aligner.result)
```

