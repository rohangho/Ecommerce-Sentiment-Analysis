## 8. Aspect-Based Sentiment Analysis with BERTopic
#Run BERTopic on the combined dataset to extract topics (aspects) for the Web UI.
from bertopic import BERTopic
import pandas as pd
import os

combined_df = pd.read_csv("Ecommerce_dataset/combined_data.csv")
combined_df['reviews.text'] = combined_df['reviews.text'].fillna('').astype(str)
docs = combined_df[combined_df['reviews.text'].str.strip() != '']['reviews.text'].tolist()
indices = combined_df[combined_df['reviews.text'].str.strip() != ''].index

print("Training BERTopic model...")
topic_model = BERTopic(language="english", calculate_probabilities=False, verbose=True)
topics, _ = topic_model.fit_transform(docs)

combined_df.loc[indices, 'topic'] = topics
topic_info = topic_model.get_topic_info()
topic_info['Name'] = topic_info['Name'].apply(lambda x: "General/Mixed" if x.startswith('-1_') else x.split('_', 1)[-1].replace('_', ', '))

topic_info.to_csv("deploy_models/bertopic_info.csv", index=False)
combined_df.to_csv("Ecommerce_dataset/combined_data_with_topics.csv", index=False)

os.makedirs("deploy_models/bertopic_model", exist_ok=True)
topic_model.save("deploy_models/bertopic_model", serialization="safetensors", save_ctfidf=True)
print("BERTopic modeling complete.")

