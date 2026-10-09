"""Text preprocessing shared by the notebook and the command-line chatbot.

Training and prediction must clean text in exactly the same way, so both
import their cleaning functions from this file.
"""

import re

import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

_NLTK_PACKAGES = {
    "stopwords": "corpora/stopwords",
    "wordnet": "corpora/wordnet",
    "omw-1.4": "corpora/omw-1.4",
}


def ensure_nltk_data():
    """Download the NLTK resources used here if they are not installed yet."""
    for package, resource in _NLTK_PACKAGES.items():
        try:
            nltk.data.find(resource)
        except LookupError:
            try:
                nltk.data.find(resource + ".zip")
            except LookupError:
                nltk.download(package, quiet=True)


ensure_nltk_data()
STOP_WORDS = set(stopwords.words("english"))
LEMMATIZER = WordNetLemmatizer()

# Phrases that reveal the *source* of an article rather than its content.
# In the Kaggle dataset almost every real article starts with a Reuters
# dateline ("WASHINGTON (Reuters) - ..."), and many fake articles end with
# image credits. A classifier can reach high accuracy just by spotting these,
# so they are removed for the "source markers removed" experiment.
_DATELINE = re.compile(r"^[^\n]{0,150}?\(reuters\)\s*[-–—]\s*", re.IGNORECASE)
_SOURCE_MARKERS = re.compile(
    r"\breuters\b"
    r"|featured image (?:via|credit)[^.\n]*"
    r"|(?:photo|image) by [^.\n]*"
    r"|getty images"
    r"|21st century wire"
    r"|pic\.twitter\.com/\S+",
    re.IGNORECASE,
)


def remove_source_markers(text):
    """Remove datelines, agency names and image credits from raw text."""
    text = _DATELINE.sub("", str(text))
    return _SOURCE_MARKERS.sub(" ", text)


def clean_text(text):
    """Lowercase, keep letters only, remove stopwords and lemmatize."""
    text = re.sub(r"http\S+|www\.\S+", " ", str(text))  # URLs
    text = re.sub(r"[^a-zA-Z\s]", " ", text).lower()     # letters only
    words = [w for w in text.split() if w not in STOP_WORDS]
    return " ".join(LEMMATIZER.lemmatize(w) for w in words)


def preprocess(text, remove_markers=True):
    """Full preprocessing used for the final model."""
    if remove_markers:
        text = remove_source_markers(text)
    return clean_text(text)
