"""Command-line chatbot: paste a news article and get a prediction.

For each article it prints
  * the prediction (real or fake) with the model's probability,
  * a short summary (DistilBART), and
  * the sentiment of the text (DistilBERT).

Run the notebook first so that models/fake_news_pipeline.joblib exists, then:

    python chatbot.py               # with summary and sentiment
    python chatbot.py --no-extras   # prediction only, no large downloads
"""

import argparse
from pathlib import Path

import joblib

from text_utils import preprocess

MODEL_PATH = Path(__file__).parent / "models" / "fake_news_pipeline.joblib"
SUMMARY_MODEL = "sshleifer/distilbart-cnn-12-6"
SENTIMENT_MODEL = "distilbert/distilbert-base-uncased-finetuned-sst-2-english"


def load_extras():
    """Load the summarization and sentiment models.

    The models are loaded directly instead of through pipeline(), because the
    "summarization" pipeline was removed in transformers v5.
    """
    import torch
    from transformers import (AutoModelForSeq2SeqLM, AutoModelForSequenceClassification,
                              AutoTokenizer, logging)

    logging.set_verbosity_error()
    sum_tok = AutoTokenizer.from_pretrained(SUMMARY_MODEL)
    sum_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARY_MODEL)
    sent_tok = AutoTokenizer.from_pretrained(SENTIMENT_MODEL)
    sent_model = AutoModelForSequenceClassification.from_pretrained(SENTIMENT_MODEL)

    def summarize(text):
        inputs = sum_tok(text, return_tensors="pt", truncation=True, max_length=1024)
        with torch.no_grad():
            ids = sum_model.generate(**inputs, max_length=100, min_length=30,
                                     num_beams=4, early_stopping=True)
        return sum_tok.decode(ids[0], skip_special_tokens=True).strip()

    def sentiment(text):
        inputs = sent_tok(text, return_tensors="pt", truncation=True, max_length=512)
        with torch.no_grad():
            probs = sent_model(**inputs).logits.softmax(dim=-1)[0]
        best = int(probs.argmax())
        return sent_model.config.id2label[best].lower(), float(probs[best])

    return summarize, sentiment


def classify(model, text):
    """Return ('Real' or 'Fake', probability of that label)."""
    proba = model.predict_proba([preprocess(text)])[0]
    label = int(proba.argmax())          # 0 = real, 1 = fake
    return ("Fake" if label == 1 else "Real"), float(proba[label])


def main():
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--no-extras", action="store_true",
                        help="skip summarization and sentiment analysis")
    args = parser.parse_args()

    if not MODEL_PATH.exists():
        raise SystemExit(f"Model not found at {MODEL_PATH}. Run the notebook first.")
    model = joblib.load(MODEL_PATH)
    summarizer = sentiment = None
    if not args.no_extras:
        print("Loading summarization and sentiment models...")
        summarizer, sentiment = load_extras()

    print("Fake News Chatbot. Paste an article and press Enter. Type 'exit' to quit.")
    while True:
        text = input("\nArticle text: ").strip()
        if text.lower() in {"exit", "quit"}:
            break
        if len(text.split()) < 20:
            print("Please paste at least a few sentences; short text gives unreliable results.")
            continue

        label, confidence = classify(model, text)
        print(f"Prediction: {label} ({confidence:.0%})")
        if summarizer is not None:
            print("Summary:", summarizer(text))
            mood, score = sentiment(text)
            print(f"Sentiment: {mood} ({score:.0%})")


if __name__ == "__main__":
    main()
