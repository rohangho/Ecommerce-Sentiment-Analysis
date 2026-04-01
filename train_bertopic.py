import os
import pandas as pd
from bertopic import BERTopic

def process_combined_data():
    print("Loading datasets...")
    train_df = pd.read_csv("Ecommerce_dataset/train_data.csv")
    
    # Optional datasets (might not exist or might be empty)
    test_hidden_path = "Ecommerce_dataset/test_data_hidden.csv"
    test_pred_path = "Ecommerce_dataset/test_data_predicted.csv"
    
    dfs_to_concat = [train_df]
    
    if os.path.exists(test_hidden_path):
        test_hidden_df = pd.read_csv(test_hidden_path)
        dfs_to_concat.append(test_hidden_df)
        
    if os.path.exists(test_pred_path):
        test_pred_df = pd.read_csv(test_pred_path)
        if 'predicted_sentiment' in test_pred_df.columns:
            test_pred_df.rename(columns={'predicted_sentiment': 'sentiment'}, inplace=True)
        dfs_to_concat.append(test_pred_df)

    common_cols = ['name', 'reviews.text', 'reviews.title', 'sentiment', 'topic']
    processed_dfs = []
    
    for df in dfs_to_concat:
        cols_present = [c for c in common_cols if c in df.columns]
        processed_dfs.append(df[cols_present])

    combined_df = pd.concat(processed_dfs, ignore_index=True)
    combined_df.to_csv("Ecommerce_dataset/combined_data.csv", index=False)
    print(f"Combined data saved to Ecommerce_dataset/combined_data.csv with {len(combined_df)} rows.")

    return combined_df

def run_bertopic():
    combined_df = process_combined_data()
    
    print("Preparing documents for BERTopic...")
    combined_df['reviews.text'] = combined_df['reviews.text'].fillna('').astype(str)
    
    # Filter out empty reviews
    mask = combined_df['reviews.text'].str.strip() != ''
    docs = combined_df[mask]['reviews.text'].tolist()
    indices = combined_df[mask].index

    print(f"Training BERTopic model on {len(docs)} documents...")
    # Add thread constraints directly in script to prevent Mac/Docker hanging
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    topic_model = BERTopic(language="english", calculate_probabilities=False, verbose=True)
    topics, _ = topic_model.fit_transform(docs)

    print("Extracting topics & saving model...")
    # Add a column for topic in the combined dataframe
    combined_df['topic'] = -1
    combined_df.loc[indices, 'topic'] = topics
    
    topic_info = topic_model.get_topic_info()
    topic_info['Name'] = topic_info['Name'].apply(
        lambda x: "General/Mixed" if str(x).startswith('-1_') else str(x).split('_', 1)[-1].replace('_', ', ')
    )

    os.makedirs("deploy_models", exist_ok=True)
    topic_info.to_csv("deploy_models/bertopic_info.csv", index=False)
    combined_df.to_csv("Ecommerce_dataset/combined_data_with_topics.csv", index=False)

    os.makedirs("deploy_models/bertopic_model", exist_ok=True)
    topic_model.save("deploy_models/bertopic_model", serialization="safetensors", save_ctfidf=True)
    print("BERTopic modeling complete.")

if __name__ == "__main__":
    run_bertopic()
