import os
import json
import pandas as pd
from bertopic import BERTopic
from bertopic.representation import KeyBERTInspired, MaximalMarginalRelevance
from sklearn.feature_extraction.text import CountVectorizer, ENGLISH_STOP_WORDS

SEED_TOPICS_FILE = "seed_topics.json"

# ─────────────────────────────────────────────────────────────────────────────
# 1.  SEED TOPICS  — loaded dynamically from seed_topics.json
#     Edit that file to add/remove/rename aspects without touching code.
#     BERTopic is SEMI-SUPERVISED: it will also discover new topics beyond
#     the seeds if the data contains aspects you haven't listed.
# ─────────────────────────────────────────────────────────────────────────────
# Hardcoded fallback (used only if seed_topics.json is missing)
_FALLBACK_SEEDS = [
    ["battery", "battery life", "charge", "charging", "charger", "mah", "power bank"],
    ["screen", "display", "resolution", "brightness", "touchscreen", "lcd", "retina", "hd"],
    ["ram", "memory", "gb ram", "ddr", "multitasking"],
    ["ssd", "storage", "hard drive", "disk", "space", "internal storage", "sd card", "micro sd"],
    ["camera", "photos", "photo", "picture", "megapixel", "lens", "selfie", "video recording"],
    ["speaker", "speakers", "sound", "audio", "bass", "volume", "headphone", "headphones"],
    ["processor", "cpu", "speed", "fast", "performance", "quad core", "lag", "slow"],
    ["wifi", "bluetooth", "wireless", "connectivity", "internet", "connection", "signal"],
    ["keyboard", "keys", "typing", "trackpad", "touchpad", "mouse"],
    ["build", "design", "weight", "lightweight", "slim", "portable", "durable", "sturdy"],
    ["software", "apps", "app", "operating system", "alexa", "update", "firmware"],
    ["price", "value", "affordable", "cheap", "expensive", "worth", "deal"],
    ["setup", "install", "installation", "easy setup", "configuration", "pairing"],
    ["warranty", "return", "refund", "defective", "broken", "replacement", "customer service"],
]


def load_seed_topics():
    """Load seed topics from seed_topics.json.
    
    Returns a list of lists (what BERTopic expects for seed_topic_list).
    If the JSON file is missing, falls back to _FALLBACK_SEEDS.
    """
    if not os.path.exists(SEED_TOPICS_FILE):
        print(f"WARNING: {SEED_TOPICS_FILE} not found — using hardcoded fallback seeds.")
        return _FALLBACK_SEEDS

    with open(SEED_TOPICS_FILE, "r") as f:
        data = json.load(f)

    seeds = [aspect["keywords"] for aspect in data["aspects"]]
    names = [aspect["name"] for aspect in data["aspects"]]
    print(f"Loaded {len(seeds)} seed aspects from {SEED_TOPICS_FILE}:")
    for name, kws in zip(names, seeds):
        print(f"  • {name}: {', '.join(kws[:5])}{'...' if len(kws) > 5 else ''}")
    return seeds

# ─────────────────────────────────────────────────────────────────────────────
# 2.  STOPWORDS — aggressively strip non-aspect filler from topic labels
# ─────────────────────────────────────────────────────────────────────────────
EXTRA_STOPWORDS = [
    # time / frequency
    "last", "hrs", "hours", "days", "weeks", "months", "year", "years", "time",
    "times", "ago", "since", "now", "still", "yet", "already", "always", "never",
    # attention / discourse
    "attention", "note", "please", "ps", "fyi",
    # generic verbs
    "get", "got", "getting", "use", "used", "using", "uses", "make", "made",
    "come", "came", "go", "went", "see", "look", "looked", "feel", "felt",
    "try", "tried", "seem", "seems", "seemed", "think", "thought", "know",
    "let", "need", "needs", "needed", "want", "wanted", "take", "took",
    "keep", "kept", "put", "give", "gave", "show", "found", "find",
    "say", "said", "tell", "told", "ask", "asked", "called", "run", "running",
    "set", "bought", "buy", "buying", "purchase", "purchased", "order", "ordered",
    # generic adjectives / adverbs
    "really", "very", "quite", "just", "also", "even", "much", "many",
    "little", "bit", "lot", "lots", "good", "great", "nice", "bad", "ok", "okay",
    "well", "right", "sure", "true", "better", "best", "new", "old", "big",
    "small", "long", "short", "high", "low", "easy", "hard", "large",
    "amazing", "awesome", "excellent", "wonderful", "perfect", "terrible",
    "horrible", "worst", "love", "loved", "loves", "hate", "hated", "happy",
    "glad", "sorry", "disappointed", "exciting", "beautiful", "pretty",
    # pronouns / articles / prepositions
    "it", "its", "this", "that", "these", "those", "they", "them", "their",
    "we", "our", "my", "me", "i", "he", "she", "you", "your", "his", "her",
    "the", "a", "an", "and", "or", "but", "if", "so", "to", "of", "for",
    "in", "on", "at", "by", "with", "from", "up", "out", "off", "do", "does",
    "did", "not", "no", "yes", "was", "were", "is", "are", "am", "be", "been",
    "has", "have", "had", "would", "could", "should", "will", "can", "may",
    "might", "shall", "than", "then", "when", "while", "where", "how", "what",
    "which", "who", "whom", "why", "some", "any", "all", "most", "more",
    "less", "only", "own", "same", "other", "another", "each", "every",
    "both", "few", "such", "too", "about", "after", "before", "between",
    "here", "there", "again", "once", "further",
    # common reviews / e-commerce filler
    "product", "item", "items", "things", "thing", "stuff", "way", "able",
    "review", "star", "stars", "rating", "amazon",
    "received", "delivery", "pack", "package", "arrived", "shipping", "ship",
    "day", "month", "week",
    # people / demographics (reviews often mention gift recipients)
    "daughter", "son", "wife", "husband", "boyfriend", "girlfriend",
    "mother", "father", "mom", "dad", "kid", "kids", "child", "children",
    "grandson", "granddaughter", "teenage", "teenager", "toddler",
    "christmas", "gift", "birthday", "present", "family",
    # vague tech words that aren't real aspects
    "model", "version", "device", "unit", "one", "two", "three", "four", "five",
    "first", "second", "tablet", "phone", "laptop", "computer", "like",
    "beautifully", "perfectly", "absolutely", "definitely", "totally",
]


def build_vectorizer():
    """CountVectorizer tuned for aspect keywords."""
    stop_words = list(ENGLISH_STOP_WORDS) + EXTRA_STOPWORDS
    return CountVectorizer(
        stop_words=stop_words,
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.80,
        token_pattern=r"(?u)\b[a-zA-Z][a-zA-Z]{2,}\b",  # words with 3+ chars
    )


def build_representation():
    """KeyBERT picks semantically relevant words; MMR adds diversity."""
    return [
        KeyBERTInspired(top_n_words=10),
        MaximalMarginalRelevance(diversity=0.3),
    ]


def clean_topic_name(raw: str) -> str:
    """'3_battery_charge_life' → 'battery, charge, life'"""
    if str(raw).startswith("-1_"):
        return "General/Mixed"
    parts = str(raw).split("_", 1)
    label = parts[-1] if len(parts) > 1 else parts[0]
    tokens = [t.strip() for t in label.split("_") if t.strip()]
    blocked = set(w.lower() for w in EXTRA_STOPWORDS)
    tokens = [t for t in tokens if t.lower() not in blocked and len(t) > 2]
    return ", ".join(tokens) if tokens else "General/Mixed"


# ─────────────────────────────────────────────────────────────────────────────
# DATA
# ─────────────────────────────────────────────────────────────────────────────
def process_combined_data():
    print("Loading datasets...")
    train_df = pd.read_csv("Ecommerce_dataset/train_data.csv")

    test_hidden_path = "Ecommerce_dataset/test_data_hidden.csv"
    test_pred_path   = "Ecommerce_dataset/test_data_predicted.csv"

    dfs_to_concat = [train_df]
    if os.path.exists(test_hidden_path):
        dfs_to_concat.append(pd.read_csv(test_hidden_path))
    if os.path.exists(test_pred_path):
        test_pred_df = pd.read_csv(test_pred_path)
        if "predicted_sentiment" in test_pred_df.columns:
            test_pred_df.rename(columns={"predicted_sentiment": "sentiment"}, inplace=True)
        dfs_to_concat.append(test_pred_df)

    common_cols = ["name", "reviews.text", "reviews.title", "sentiment", "topic"]
    processed_dfs = [df[[c for c in common_cols if c in df.columns]] for df in dfs_to_concat]

    combined_df = pd.concat(processed_dfs, ignore_index=True)
    combined_df.to_csv("Ecommerce_dataset/combined_data.csv", index=False)
    print(f"Combined data saved with {len(combined_df)} rows.")
    return combined_df


# ─────────────────────────────────────────────────────────────────────────────
# TRAINING
# ─────────────────────────────────────────────────────────────────────────────
def run_bertopic():
    combined_df = process_combined_data()

    print("Preparing documents for BERTopic...")
    combined_df["reviews.text"] = combined_df["reviews.text"].fillna("").astype(str)
    mask    = combined_df["reviews.text"].str.strip() != ""
    docs    = combined_df[mask]["reviews.text"].tolist()
    indices = combined_df[mask].index

    os.environ["OMP_NUM_THREADS"]        = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    seed_topics = load_seed_topics()
    print(f"Training Guided BERTopic on {len(docs)} documents with {len(seed_topics)} seed aspects...")
    topic_model = BERTopic(
        language="english",
        calculate_probabilities=False,
        vectorizer_model=build_vectorizer(),
        representation_model=build_representation(),
        seed_topic_list=seed_topics,       # ← loaded from seed_topics.json
        top_n_words=6,
        min_topic_size=15,                 # allow smaller, more specific aspect clusters
        nr_topics="auto",                  # merge near-duplicates
        verbose=True,
    )
    topics, _ = topic_model.fit_transform(docs)

    # ── Post-process ──────────────────────────────────────────────────────
    print("Extracting topics & saving model...")
    combined_df["topic"] = -1
    combined_df.loc[indices, "topic"] = topics

    topic_info = topic_model.get_topic_info()
    topic_info["Name"] = topic_info["Name"].apply(clean_topic_name)

    os.makedirs("deploy_models", exist_ok=True)
    topic_info.to_csv("deploy_models/bertopic_info.csv", index=False)
    combined_df.to_csv("Ecommerce_dataset/combined_data_with_topics.csv", index=False)

    os.makedirs("deploy_models/bertopic_model", exist_ok=True)
    topic_model.save(
        "deploy_models/bertopic_model",
        serialization="safetensors",
        save_ctfidf=True,
    )

    # Print topic quality summary
    print("\n=== TOPIC SUMMARY ===")
    for _, row in topic_info.iterrows():
        if row["Topic"] == -1:
            continue
        print(f"  Topic {row['Topic']:>3d}: {row['Name']:<40s} ({row['Count']} docs)")
    print("BERTopic modeling complete.\n")


if __name__ == "__main__":
    run_bertopic()
