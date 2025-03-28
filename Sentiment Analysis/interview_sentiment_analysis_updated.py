# pip install python-docx geotext scikit-learn transformers pandas torch spacy
# python -m spacy download en_core_web_sm

import os
import re
import pandas as pd
import logging
from docx import Document
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import torch
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
import numpy as np
from geotext import GeoText
import spacy

# Load English language model for NLP
nlp = spacy.load("en_core_web_sm")

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("interview_analysis.log"),
        logging.StreamHandler()
    ]
)

# Configuration
INTERVIEWS_FOLDER = "/path/to/your/interviews"
OUTPUT_FOLDER = "/path/to/output"
QUESTION_KEYWORDS = {
    "household": ["family", "home", "children", "personal", "daily life"],
    "stakeholder": ["organization", "policy", "community", "program", "official"]
}
MAX_TOKENS = 4000
MIN_CHUNK_SIZE = 500

# Initialize analysis pipelines
device = 0 if torch.cuda.is_available() else -1
sentiment_analyzer = pipeline(
    "text-classification",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=device
)

def load_interview_text(docx_path):
    """Extract text from Word document with metadata preservation"""
    try:
        doc = Document(docx_path)
        metadata = {}
        content = []
        
        for para in doc.paragraphs:
            text = para.text.strip()
            if ":" in text and len(text.split(":")) == 2:
                key, value = text.split(":", 1)
                metadata[key.strip().lower()] = value.strip()
            elif text:
                content.append(text)
        
        return {
            "metadata": metadata,
            "content": "\n".join(content)
        }
    except Exception as e:
        logging.error(f"Error loading {docx_path}: {str(e)}")
        return {"metadata": {}, "content": ""}

def classify_interview_type(text):
    """Classify interview as household or stakeholder based on question content"""
    household_score = sum(text.lower().count(kw) for kw in QUESTION_KEYWORDS["household"])
    stakeholder_score = sum(text.lower().count(kw) for kw in QUESTION_KEYWORDS["stakeholder"])
    
    if household_score > stakeholder_score:
        return "household"
    elif stakeholder_score > household_score:
        return "stakeholder"
    return "unknown"

def extract_location(text):
    """Enhanced location extraction using spaCy NER"""
    doc = nlp(text)
    locations = []
    
    for ent in doc.ents:
        if ent.label_ in ["GPE", "LOC"]:
            locations.append(ent.text)
    
    return locations[0] if locations else "Unknown"

def classify_urban_rural(location):
    """Classify location as urban or rural (simplified example)"""
    urban_keywords = ["city", "metro", "urban"]
    if any(kw in location.lower() for kw in urban_keywords):
        return "Urban"
    return "Rural"

def chunk_interview(text, target_size=MAX_TOKENS, min_size=MIN_CHUNK_SIZE):
    """Improved chunking that respects question boundaries"""
    questions = re.split(r'\n\s*[Qq]\d*[.:]?\s*', text)  # Split by question markers
    chunks = []
    current_chunk = []
    current_length = 0
    
    for question in questions:
        if not question.strip():
            continue
            
        q_length = len(question.split())
        if current_length + q_length > target_size and current_length >= min_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = [question]
            current_length = q_length
        else:
            current_chunk.append(question)
            current_length += q_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def analyze_sentiment_chunks(chunks):
    """Enhanced sentiment analysis with chunk weighting"""
    if not chunks:
        return {"positive": 0, "neutral": 0, "negative": 0}
    
    sentiments = []
    weights = []
    
    for chunk in chunks:
        try:
            chunk_len = len(chunk.split())
            result = sentiment_analyzer(chunk[:5000])[0]
            sentiments.append(result['label'].lower())
            weights.append(chunk_len)
        except Exception as e:
            logging.warning(f"Sentiment analysis failed for chunk: {str(e)}")
            continue
    
    if not sentiments:
        return {"positive": 0, "neutral": 0, "negative": 0}
    
    # Calculate weighted proportions
    total_weight = sum(weights)
    positive = sum(w for w, s in zip(weights, sentiments) if s == "positive") / total_weight * 100
    neutral = sum(w for w, s in zip(weights, sentiments) if s == "neutral") / total_weight * 100
    negative = sum(w for w, s in zip(weights, sentiments) if s == "negative") / total_weight * 100
    
    return {
        "positive": round(positive, 2),
        "neutral": round(neutral, 2),
        "negative": round(negative, 2)
    }

def extract_key_themes(text_chunks, interview_type, n_themes=3):
    """Type-specific theme extraction"""
    try:
        # Type-specific stop words
        custom_stop = []
        if interview_type == "household":
            custom_stop = ["family", "home"]  # Common words we want to keep
        elif interview_type == "stakeholder":
            custom_stop = ["organization", "community"]
        
        vectorizer = TfidfVectorizer(
            stop_words=list(set(TfidfVectorizer.get_stop_words()) - set(custom_stop)),
            ngram_range=(1, 3),
            max_df=0.85,
            min_df=2
        )
        X = vectorizer.fit_transform(text_chunks)
        
        features = vectorizer.get_feature_names_out()
        if len(features) < n_themes:
            return ["Insufficient data"]
        
        kmeans = KMeans(n_clusters=n_themes, random_state=42)
        kmeans.fit(X)
        
        themes = []
        for i in range(kmeans.n_clusters):
            centroid = kmeans.cluster_centers_[i]
            top_words_idx = centroid.argsort()[-5:][::-1]
            themes.append(", ".join(features[idx] for idx in top_words_idx))
        
        return themes
    except Exception as e:
        logging.warning(f"Theme extraction failed: {str(e)}")
        return ["Analysis error"]

def process_interview(file_path):
    """Enhanced interview processing pipeline"""
    try:
        # Load and classify interview
        interview = load_interview_text(file_path)
        if not interview["content"].strip():
            return None
        
        interview_type = classify_interview_type(interview["content"])
        location = extract_location(interview["content"])
        urban_rural = classify_urban_rural(location)
        
        # Chunk and analyze
        chunks = chunk_interview(interview["content"])
        sentiment = analyze_sentiment_chunks(chunks)
        themes = extract_key_themes(chunks, interview_type)
        
        # Extract respondent info if available
        respondent_info = {
            "age": interview["metadata"].get("age", "unknown"),
            "gender": interview["metadata"].get("gender", "unknown"),
            "occupation": interview["metadata"].get("occupation", "unknown")
        }
        
        return {
            "filename": os.path.basename(file_path),
            "interview_type": interview_type,
            "location": location,
            "urban_rural": urban_rural,
            "word_count": len(interview["content"].split()),
            "chunks": len(chunks),
            "positive": sentiment["positive"],
            "neutral": sentiment["neutral"],
            "negative": sentiment["negative"],
            "key_themes": " | ".join(themes),
            **respondent_info
        }
    except Exception as e:
        logging.error(f"Error processing {file_path}: {str(e)}")
        return None

def generate_summary_tables(results):
    """Generate multiple summary views"""
    if not results:
        return {}
    
    df = pd.DataFrame(results)
    
    # 1. By interview type and urban/rural
    type_location = df.groupby(['interview_type', 'urban_rural']).agg({
        'positive': 'mean',
        'neutral': 'mean',
        'negative': 'mean'
    }).round(2).reset_index()
    
    # 2. Key themes by group
    themes_by_group = df.groupby('interview_type')['key_themes'].apply(
        lambda x: extract_common_themes(x, n=5)
    ).reset_index()
    
    # 3. Sentiment distribution by demographic factors
    demographic_tables = {}
    for factor in ['age', 'gender', 'occupation']:
        if factor in df.columns:
            demo_table = df.groupby([factor, 'interview_type']).agg({
                'positive': 'mean',
                'neutral': 'mean',
                'negative': 'mean'
            }).round(2).reset_index()
            demographic_tables[factor] = demo_table
    
    return {
        "by_type_location": type_location,
        "key_themes": themes_by_group,
        "demographics": demographic_tables
    }

def extract_common_themes(theme_series, n=3):
    """Improved theme consolidation"""
    all_themes = []
    for themes in theme_series:
        all_themes.extend(themes.split(" | "))
    
    if not all_themes:
        return "No common themes"
    
    theme_counts = pd.Series(all_themes).value_counts()
    return ", ".join(f'"{theme}"' for theme in theme_counts.head(n).index.tolist())

def main():
    """Enhanced main function"""
    logging.info("Starting enhanced interview analysis")
    
    # Find all interview files
    interview_files = []
    for root, _, files in os.walk(INTERVIEWS_FOLDER):
        for file in files:
            if file.endswith(".docx"):
                interview_files.append(os.path.join(root, file))
    
    if not interview_files:
        logging.error("No interview files found")
        return
    
    # Process in parallel
    results = []
    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(process_interview, path) for path in interview_files]
        for future in futures:
            result = future.result()
            if result:
                results.append(result)
    
    # Save and analyze results
    if not results:
        logging.error("No valid interviews processed")
        return
    
    raw_df = pd.DataFrame(results)
    raw_output = os.path.join(OUTPUT_FOLDER, "interview_results_enhanced.csv")
    raw_df.to_csv(raw_output, index=False)
    
    # Generate summary tables
    tables = generate_summary_tables(results)
    
    # Save all tables
    for name, table in tables.items():
        if isinstance(table, dict):
            for demo, demo_table in table.items():
                demo_output = os.path.join(OUTPUT_FOLDER, f"summary_{name}_{demo}.csv")
                demo_table.to_csv(demo_output, index=False)
        else:
            table_output = os.path.join(OUTPUT_FOLDER, f"summary_{name}.csv")
            table.to_csv(table_output, index=False)
    
    # Print key results
    print("\n=== SENTIMENT BY INTERVIEW TYPE AND LOCATION ===")
    print(tables["by_type_location"].to_string(index=False))
    
    print("\n=== KEY THEMES BY GROUP ===")
    print(tables["key_themes"].to_string(index=False))
    
    print(f"\nFull results saved to {OUTPUT_FOLDER}")

if __name__ == "__main__":
    main()
