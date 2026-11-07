"""Command-line entry point for the Week 4 analysis."""
from src.analysis import run_analysis

if __name__ == "__main__":
    result = run_analysis()
    print("Week 4 analysis completed.")
    print(f"Rows analyzed: {result['rows_cleaned']}")
    print(f"Missing values filled: {result['raw_missing_co2']}")
    print(f"Visualizations created: {result['figures_created']}")
    print("Outputs: outputs/ and data/processed/")
