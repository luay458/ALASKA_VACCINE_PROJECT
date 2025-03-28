# Install required libraries
!pip install pandas transformers geopy vaderSentiment torch concurrent-log-handler

import tarfile
import json
import os
import pandas as pd
import re
from transformers import pipeline, AutoTokenizer
from google.colab import drive
import logging
from concurrent.futures import ThreadPoolExecutor
from functools import lru_cache
import torch
from concurrent_log_handler import ConcurrentRotatingFileHandler
from collections import defaultdict
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans

# Set up logging
log_file_path = "/content/drive/MyDrive/alaska_vaccine_project/tweets_analysis.log"
handler = ConcurrentRotatingFileHandler(log_file_path, "a", 1024*1024, 5)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[handler]
)

# Mount Google Drive
drive.mount('/content/drive')

# Sentiment label mapping
LABEL_MAPPING = {
    "LABEL_0": "Negative",
    "LABEL_1": "Neutral",
    "LABEL_2": "Positive"
}

# Load sentiment analysis model & tokenizer
model_name = "cardiffnlp/twitter-roberta-base-sentiment"
device = 0 if torch.cuda.is_available() else -1
classifier = pipeline(
    "sentiment-analysis", 
    model=model_name, 
    tokenizer=model_name,
    device=device,
    truncation=True,
    max_length=512
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Define paths
tweets_tar_path = "/content/drive/MyDrive/alaska_vaccine_project/tweets/gemma-keras-gemma_1.1_instruct_2b_en-v4.tar.gz"
tweets_csv_folder = "/content/drive/MyDrive/alaska_vaccine_project/tweets/csv_tweets"
output_folder = "/content/drive/MyDrive/alaska_vaccine_project/results"

# Create output folder if it doesn't exist
os.makedirs(output_folder, exist_ok=True)

@lru_cache(maxsize=10000)
def is_in_alaska(location):
    """
    Check if a location string refers to Alaska using regex patterns.
    Caches results for performance.
    """
    if not isinstance(location, str):
        return False

    alaska_patterns = [
        r"\bAlaska\b", r"\bAK\b", 
        r"Anchorage", r"Sitka", r"Juneau", r"Fairbanks", r"Wasilla", r"Kenai",
        r"\bAK,?\s*(USA|US|United States)?\b", 
        r"\bAlaska,?\s*(USA|US|United States)?\b",
        r"\bAK,\s*\w{2,}\b",
        r"Alaskan", r"Mat-Su", r"Kodiak", r"Valdez", r"Barrow", r"Nome"
    ]

    return any(re.search(pattern, location, re.IGNORECASE) for pattern in alaska_patterns)

def classify_rural_urban(location):
    """
    Classify location as Rural, Urban, or Unknown based on keywords.
    """
    if not isinstance(location, str):
        return "Unknown"

    rural_keywords = {
        "Bethel", "Nome", "Barrow", "Kotzebue", "Wrangell", "Dillingham",
        "Rural", "Village", "Borough", "Cordova", "Valdez", "Homer", "Kenai"
    }
    
    urban_keywords = {
        "Anchorage", "Fairbanks", "Juneau", "Sitka", "Wasilla", "Urban",
        "City", "Metro", "Ketchikan", "Kodiak"
    }

    location_lower = location.lower()
    
    if any(keyword.lower() in location_lower for keyword in rural_keywords):
        return "Rural"
    if any(keyword.lower() in location_lower for keyword in urban_keywords):
        return "Urban"
    
    return "Unknown"

def truncate_text(text):
    """
    Truncate text to 512 tokens for the sentiment model.
    """
    if not isinstance(text, str):
        return ""
    
    tokens = tokenizer.encode(text, truncation=True, max_length=512)
    return tokenizer.decode(tokens, skip_special_tokens=True)

def extract_tweets_from_tar(file_path):
    """
    Extract tweets from a tar.gz file with robust error handling.
    """
    tweets = []
    try:
        with tarfile.open(file_path, "r:gz") as tar:
            for member in tar.getmembers():
                try:
                    f = tar.extractfile(member)
                    if f:
                        for line in f:
                            try:
                                # Try utf-8 first
                                tweet = json.loads(line.decode('utf-8', errors='replace'))
                                if isinstance(tweet, dict):
                                    tweets.append(tweet)
                            except json.JSONDecodeError:
                                try:
                                    # Fallback to latin-1
                                    tweet = json.loads(line.decode('latin-1', errors='replace'))
                                    if isinstance(tweet, dict):
                                        tweets.append(tweet)
                                except json.JSONDecodeError:
                                    continue
                except Exception as e:
                    logging.warning(f"Error processing member {member.name}: {str(e)}")
                    continue
    except Exception as e:
        logging.error(f"Error opening tar file {file_path}: {str(e)}")
        raise

    logging.info(f"Extracted {len(tweets)} tweets from {file_path}")
    return tweets

def load_csv_safely(file_path):
    """
    Attempt to load a CSV file with multiple encodings.
    Returns DataFrame if successful, None otherwise.
    """
    encodings = ["utf-8", "latin-1", "ISO-8859-1", "cp1252"]
    for encoding in encodings:
        try:
            df = pd.read_csv(
                file_path, 
                encoding=encoding, 
                engine="python", 
                on_bad_lines="skip",
                dtype={'user_location': str, 'description': str}
            )
            # Ensure required columns exist
            if "description" not in df.columns:
                df["description"] = ""
            if "user_location" not in df.columns:
                df["user_location"] = ""
                
            logging.info(f"Successfully read {file_path} with {encoding}")
            return df
        except Exception as e:
            logging.warning(f"Failed to read {file_path} with {encoding}: {str(e)}")
    
    logging.error(f"Skipping {file_path} - All encoding attempts failed.")
    return None

def load_tweets_from_csv_folder(folder_path):
    """
    Load tweets from all CSV files in a folder.
    """
    all_tweets = []
    if not os.path.exists(folder_path):
        logging.error(f"Folder not found: {folder_path}")
        return all_tweets

    for filename in os.listdir(folder_path):
        if filename.endswith(".csv"):
            file_path = os.path.join(folder_path, filename)
            try:
                df = load_csv_safely(file_path)
                if df is None:
                    continue

                # Standardize column names and handle missing data
                df = df.rename(columns={"description": "text"})
                df["text"] = df["text"].fillna("").astype(str)
                df["user_location"] = df["user_location"].fillna("").astype(str)

                # Convert to list of dicts
                tweets = df[["text", "user_location"]].to_dict(orient="records")
                all_tweets.extend(tweets)
                logging.info(f"Loaded {len(tweets)} tweets from {filename}")
            except Exception as e:
                logging.error(f"Error processing {filename}: {str(e)}")
                continue

    logging.info(f"Total tweets loaded from CSV folder: {len(all_tweets)}")
    return all_tweets
def extract_key_phrases(texts, n_phrases=3, n_clusters=3):
    """
    Extract key phrases from a list of texts using TF-IDF and clustering.
    Returns a list of representative phrases.
    """
    if not texts or len(texts) < 2:
        return ["Insufficient data"]
    
    try:
        # Create TF-IDF matrix
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2),
            max_df=0.8,
            min_df=2
        )
        X = vectorizer.fit_transform(texts)
        
        # Get feature names (words/phrases)
        features = vectorizer.get_feature_names_out()
        
        # If we have enough features, perform clustering
        if len(features) > n_clusters:
            # Perform K-means clustering
            kmeans = KMeans(n_clusters=min(n_clusters, len(features)), random_state=42)
            kmeans.fit(X)
            
            # Get top phrases per cluster
            common_words = []
            for i in range(kmeans.n_clusters):
                centroid = kmeans.cluster_centers_[i]
                top_words_idx = centroid.argsort()[-n_phrases:][::-1]
                common_words.append([features[idx] for idx in top_words_idx])
            
            # Flatten and deduplicate
            phrases = list(set([phrase for sublist in common_words for phrase in sublist]))
            return phrases[:n_phrases*2]  # Return more phrases for better coverage
        else:
            # Fallback: just return most common words
            sums = X.sum(axis=0)
            top_words_idx = np.array(sums).argsort()[0][-n_phrases*2:][::-1]
            return [features[idx] for idx in top_words_idx]
    except Exception as e:
        logging.warning(f"Error in phrase extraction: {str(e)}")
        return ["Analysis failed"]

def generate_results_table(df):
    """
    Generate a comprehensive results table with sentiment percentages and key phrases.
    Returns a DataFrame in the requested format.
    """
    if df.empty:
        return pd.DataFrame()
    
    # Create cross-tabulation of sentiment vs rural/urban
    cross_tab = pd.crosstab(
        index=df['sentiment'],
        columns=df['rural_urban'],
        normalize='columns'
    ).mul(100).round(2)
    
    # Get unique sentiment and location categories
    sentiments = df['sentiment'].unique()
    locations = ['Urban', 'Rural', 'Unknown']
    
    # Prepare to collect results
    results = []
    
    # Analyze each combination
    for sentiment in sentiments:
        for location in locations:
            # Filter data for this combination
            mask = (df['sentiment'] == sentiment) & (df['rural_urban'] == location)
            subset = df[mask]
            
            if len(subset) == 0:
                continue
                
            # Get percentage from cross tab
            percentage = cross_tab.loc[sentiment, location]
            
            # Extract key phrases
            texts = subset['text'].tolist()
            key_phrases = extract_key_phrases(texts)
            
            # Format phrases as quoted strings
            formatted_phrases = ', '.join(f'"{phrase}"' for phrase in key_phrases)
            
            # Add to results
            results.append({
                'Sentiment Category': sentiment,
                'Urban/Rural': location,
                'Percentage of Posts': f"{percentage}%",
                'Key Phrases Identified': formatted_phrases
            })
    
    # Create DataFrame and sort it
    results_df = pd.DataFrame(results)
    if not results_df.empty:
        results_df = results_df.sort_values(['Sentiment Category', 'Urban/Rural'])
    
    return results_df


def process_tweet_batch(tweet_batch):
    """
    Process a batch of tweets in parallel.
    """
    batch_results = []
    for tweet in tweet_batch:
        try:
            text = tweet.get("text", "").strip()
            location = tweet.get("user_location", "").strip()
            
            if not text or not is_in_alaska(location):
                continue

            truncated_text = truncate_text(text)
            if not truncated_text:
                continue

            # Classify rural/urban
            rural_urban = classify_rural_urban(location)

            batch_results.append({
                "text": truncated_text,
                "location": location,
                "rural_urban": rural_urban
            })
        except Exception as e:
            logging.warning(f"Error processing tweet: {str(e)}")
            continue
    
    return batch_results

def analyze_sentiment_batch(text_batch):
    """
    Analyze sentiment for a batch of texts.
    """
    try:
        results = classifier(text_batch)
        return [LABEL_MAPPING.get(r["label"], r["label"]) for r in results]
    except Exception as e:
        logging.error(f"Error in sentiment analysis: {str(e)}")
        return ["Error"] * len(text_batch)

def process_tweets(file_path, data_source="csv", batch_size=32, save_interval=1000):
    """
    Main function to process tweets from either tar or csv source.
    Returns both the raw data and results table.
    """
    logging.info(f"Starting tweet processing with source: {data_source}")
    
    # Load tweets based on source
    if data_source == "tar":
        tweets = extract_tweets_from_tar(file_path)
    elif data_source == "csv":
        tweets = load_tweets_from_csv_folder(file_path)
    else:
        raise ValueError("Invalid data source. Use 'tar' or 'csv'.")

    if not tweets:
        logging.warning("No tweets found or loaded.")
        return pd.DataFrame()

    # Process tweets in parallel batches
    alaska_tweets = []
    try:
        with ThreadPoolExecutor() as executor:
            # Split tweets into batches for parallel processing
            tweet_batches = [tweets[i:i + batch_size] for i in range(0, len(tweets), batch_size)]
            
            # Process location filtering and text truncation in parallel
            for batch_result in executor.map(process_tweet_batch, tweet_batches):
                alaska_tweets.extend(batch_result)
                
                # Save intermediate results periodically
                if len(alaska_tweets) % save_interval == 0:
                    temp_df = pd.DataFrame(alaska_tweets)
                    temp_path = os.path.join(output_folder, f"temp_results_{len(alaska_tweets)}.csv")
                    temp_df.to_csv(temp_path, index=False)
                    logging.info(f"Saved intermediate results: {temp_path}")
    except Exception as e:
        logging.error(f"Error in parallel processing: {str(e)}")
        raise

    if not alaska_tweets:
        logging.warning("No Alaska tweets found after filtering.")
        return pd.DataFrame()

    logging.info(f"Found {len(alaska_tweets)} Alaska tweets. Starting sentiment analysis...")

    # Batch sentiment analysis
    text_batches = [
        [t["text"] for t in alaska_tweets[i:i + batch_size]] 
        for i in range(0, len(alaska_tweets), batch_size)
    ]
    
    sentiment_results = []
    try:
        with ThreadPoolExecutor() as executor:
            for batch_sentiments in executor.map(analyze_sentiment_batch, text_batches):
                sentiment_results.extend(batch_sentiments)
    except Exception as e:
        logging.error(f"Error in batch sentiment analysis: {str(e)}")
        raise

    # Combine results
    for i, tweet in enumerate(alaska_tweets):
        if i < len(sentiment_results):
            tweet["sentiment"] = sentiment_results[i]
        else:
            tweet["sentiment"] = "Error"

    # Create final DataFrame
    result_df = pd.DataFrame(alaska_tweets)
    
    # Save final results
    output_path = os.path.join(output_folder, f"tweets_sentiment_{data_source}.csv")
    result_df.to_csv(output_path, index=False)
    logging.info(f"Analysis complete. Results saved to: {output_path}")

    # Create final DataFrame
    result_df = pd.DataFrame(alaska_tweets)
    
    # Generate results table
    results_table = generate_results_table(result_df)
    
    # Save both files
    raw_output_path = os.path.join(output_folder, f"tweets_sentiment_{data_source}.csv")
    table_output_path = os.path.join(output_folder, f"results_table_{data_source}.csv")
    
    result_df.to_csv(raw_output_path, index=False)
    results_table.to_csv(table_output_path, index=False)
    
    logging.info(f"Analysis complete. Results saved to:\n- {raw_output_path}\n- {table_output_path}")
    
    return result_df, results_table

# Main execution
if __name__ == "__main__":
    try:
        # ⚡ SWITCH DATASET HERE:
        data_source = "csv"  # or "tar"
        file_path = tweets_csv_folder if data_source == "csv" else tweets_tar_path

        logging.info(f"Starting analysis with {data_source} dataset")
        df, results_table = process_tweets(file_path, data_source)

        if not df.empty:
            print(f"Analysis complete. Found {len(df)} Alaska tweets.")
            print("\n=== Results Table ===")
            print(results_table.to_string(index=False))
        else:
            print("No valid Alaska tweets were found.")
    except Exception as e:
        logging.error(f"Fatal error in main execution: {str(e)}")
        raise
