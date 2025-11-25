import pandas as pd
import spacy

# Load spaCy model
nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])

# Helper functions
def clean_spacy(doc):
    """Lemmatize, lowercase, and remove stopwords & punctuation."""
    return " ".join(
        token.lemma_.lower().strip()
        for token in doc
        if not token.is_stop and not token.is_punct
    )

def assign_sentiment(rating: float) -> str:
    """Assign sentiment label based on numeric rating."""
    if rating >= 4:
        return "positive"
    if rating == 3:
        return "neutral"
    return "negative"

# Load dataset
df = pd.read_csv("reviews_dataset.csv")

# Clean review text
texts = df["review_full"].astype(str).tolist()
df["clean_text"] = [
    clean_spacy(doc) for doc in nlp.pipe(texts)
]

# Clean up and label
df = df.dropna(subset=["clean_text", "rating_review"])
df["sentiment"] = df["rating_review"].apply(assign_sentiment)

# Save result
output_path = "preprocessed_reviews.csv"
df.to_csv(output_path, index=False)

print(f"✅ Preprocessing complete. File saved as '{output_path}'")
