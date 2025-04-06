import pandas as pd
from pathlib import Path
from urllib.request import urlopen


def preprocess_github_spam() -> None:
    """Clean and prepare the spam dataset and save it in data/processed"""
    Path("dataset/raw").mkdir(parents=True, exist_ok=True)
    Path("dataset/processed").mkdir(parents=True, exist_ok=True)

    url = "https://raw.githubusercontent.com/AdityakChauhan/spam_detection/main/dataset/dataset.csv"
    
    try:
        # Read directly from URL
        df = pd.read_csv(url)
        
        # Save raw data
        df.to_csv("dataset/raw/dataset.csv", index=False)
        
        df = df.rename(columns={'spam': 'label'})
        
        df = df[['text', 'label']]
        
        df = df.fillna("")  # Replace any NaN values
        df = df.dropna()  # Remove any rows that might still have NaN values after filling
        df = df.drop_duplicates()  # Remove duplicates
        
        # Save processed data
        df.to_csv("dataset/processed/dataset.csv", index=False)
        print(f"Processed data saved with {len(df)} records")
        print(df.head)
        
    except Exception as e:
        print(f"Error processing GitHub spam dataset: {e}")

if __name__ == "__main__":
    preprocess_github_spam()
    