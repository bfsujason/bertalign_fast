import os
import re

import pysbd
import pycld2 as cld2

from sentence_splitter import SentenceSplitter

# Maximum number of characters sampled for language detection.
_LANG_DETECT_SAMPLE_LENGTH = 500

# ISO 639-1 codes and language names for all supported languages.
SUPPORTED_LANGUAGES = {
    "ar": "Arabic",
    "bg": "Bulgarian",
    "ca": "Catalan",
    "cs": "Czech",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "es": "Spanish",
    "fa": "Persian",
    "fi": "Finnish",
    "fr": "French",
    "hu": "Hungarian",
    "hi": "Hindi",
    "hy": "Armenian",
    "it": "Italian",
    "ja": "Japanese",
    "lt": "Lithuanian",
    "lv": "Latvian",
    "mr": "Marathi",
    "my": "Burmese",
    "nl": "Dutch",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "sk": "Slovak",
    "sl": "Slovenian",
    "sv": "Swedish",
    "tr": "Turkish",
    "ur": "Urdu",
    "zh": "Chinese",
}


def clean_text(text):
    """Normalise whitespace and remove blank lines from raw document text.

    Each non-empty line is stripped and has internal whitespace runs collapsed
    to a single space. Blank lines are discarded entirely. The result is
    rejoined with newline separators so that paragraph boundaries are preserved
    for downstream sentence splitting.

    Args:
        text: Raw document string.

    Returns:
        Cleaned string with one paragraph per line.
    """
    cleaned_lines = []
    for line in text.strip().splitlines():
        line = line.strip()
        if line:
            cleaned_lines.append(re.sub(r'\s+', ' ', line))
    return "\n".join(cleaned_lines)


def detect_lang(text):
    """Detect the language of text and validate that it is supported.

    Uses pycld2 on a short chunk of the text. Chinese language
    variants (zh-Hant, zh-Hans, etc.) are normalised to 'zh'.

    Args:
        text: Input document string (only the first few hundred characters
              are examined).

    Returns:
        Human-readable language name (e.g. 'English', 'Chinese').

    Raises:
        ValueError: If the detected language is not in `SUPPORTED_LANGUAGES`.
    """
    sample = text[:_LANG_DETECT_SAMPLE_LENGTH]
    _, _, details = cld2.detect(sample)
    lang_name, lang_code, _, _ = details[0]

    # pycld2 may return region-specific codes like 'zh-Hant'.
    if lang_code.startswith("zh"):
        lang_code = "zh"

    if lang_code not in SUPPORTED_LANGUAGES:
        raise ValueError(
            f"{lang_name} ({lang_code}) is not supported. "
            f"Supported languages: {', '.join(sorted(SUPPORTED_LANGUAGES))}"
        )

    return lang_code, SUPPORTED_LANGUAGES[lang_code]
    
def split_sents(text, lang_code):
    if lang_code == "zh":
        sents = _split_zh_sents(text)
    elif lang_code in ["ar", "bg", "fa", "hi", "hy", "ja", "mr", "my", "ur"]: # pysbd
        splitter = pysbd.Segmenter(language=lang_code, clean=False)
        sents = splitter.segment(text)
        sents = [sent.strip() for sent in sents]
    else: # ssplit
        splitter = SentenceSplitter(language=lang_code)
        sents = splitter.split(text=text)
        sents = [sent.strip() for sent in sents]
    return sents
    
def _split_zh_sents(text, limit=1000):
        sent_list = []
        text = re.sub('(?P<quotation_mark>([。？！](?![”’"\'）])))', r'\g<quotation_mark>\n', text)
        text = re.sub('(?P<quotation_mark>([。？！]|…{1,2})[”’"\'）])', r'\g<quotation_mark>\n', text)

        sent_list_ori = text.splitlines()
        for sent in sent_list_ori:
            sent = sent.strip()
            if not sent:
                continue
            else:
                while len(sent) > limit:
                    temp = sent[0:limit]
                    sent_list.append(temp)
                    sent = sent[limit:]
                sent_list.append(sent)

        return sent_list
