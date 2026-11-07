from pathlib import Path
from src.preprocessing import run_preprocessing

if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    result = run_preprocessing(root)
    print("WEEK 6 DATA PREPROCESSING COMPLETED")
    print("Raw rows:", len(result['raw']))
    print("Rows after duplicate removal:", len(result['validated']))
    print("Duplicates removed:", result['duplicate_report']['duplicates_removed'])
    print("Invalid values converted to missing:", sum(result['domain_report'].values()))
    print("Train rows:", len(result['X_train']))
    print("Test rows:", len(result['X_test']))
    print("Encoded feature count:", result['outputs']['standard']['feature_count'])
    print("Outputs saved under data/processed, models and outputs/")
