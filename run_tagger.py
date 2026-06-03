#!/usr/bin/env python
#%%
"""
Tag text data with clause-level genericity labels.

This script applies the Anecdotal Discourse Classifier to tag each clause
in the input text data with genericity labels, then saves the results.
"""
import os
import argparse
import pandas as pd
from pathlib import Path
from tqdm.auto import tqdm
from Anecedotal_Discourse_Classifier_Multitext.pipeline import run_pipeline
#%%

def main():
    parser = argparse.ArgumentParser(
        description="Tag text data with clause-level genericity labels using the Anecdotal Discourse Classifier."
    )
    parser.add_argument(
        "--input",
        "-i",
        type=str,
        required=True,
        help="Path to input CSV file with a 'text' column (or 'generated_text', which will be renamed)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=str,
        default=None,
        help="Path to output tagged CSV file (default: input_filename_tagged.csv in ./data/)",
    )
    parser.add_argument(
        "--sample-data",
        type=float,
        default=None,
        help="Fraction of data to sample for tagging (default: None, use all data)",
    )

    args = parser.parse_args()

    input_path = Path(args.input).expanduser()

    # Set default output path based on input filename if not specified
    if args.output is None:
        output_path = Path("./data") / f"{input_path.stem}_tagged.csv.gz"
    else:
        output_path = Path(args.output).expanduser()

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Loading data from: {input_path}")

    # Load the dataset
    try:
        data = pd.read_csv(input_path)
    except FileNotFoundError:
        print(f"Error: Input file not found: {input_path}")
        return 1
    except Exception as e:
        print(f"Error loading input file: {e}")
        return 1
    # Sample data if specified
    if args.sample_data is not None:
        data = data.sample(frac=args.sample_data, random_state=42).reset_index(
            drop=True
        )
        print(f"Sampled {len(data)} rows ({args.sample_data * 100:.1f}%) for tagging")

    # Ensure 'text' column exists, or rename 'generated_text' if present
    if "text" in data.columns:
        pass  # 'text' column exists, do nothing
    elif "generated_text" in data.columns:
        data = data.rename(columns={"generated_text": "text"})
        print("Renamed 'generated_text' column to 'text'")
    else:
        print(
            f"Error: Input CSV must contain a 'text' column. Found columns: {data.columns.tolist()}"
        )
        return 1

    print(f"Loaded {len(data)} rows")
    print(f"Tagging clauses with genericity labels...")

    tqdm.pandas()

    # Apply the tagging pipeline
    data["clause2labels"] = data["text"].progress_apply(lambda x: run_pipeline(x)[1])

    # Save the tagged data in compressed csv.gz
    print(f"Saving tagged data to: {output_path}")
    data.to_csv(output_path, index=False, compression="gzip")

    print(f"✓ Successfully tagged and saved {len(data)} texts")
    return 0

#%%
if __name__ == "__main__":
    exit(main())
