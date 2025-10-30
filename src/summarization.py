
import pandas as pd
from transformers import pipeline
from tqdm import tqdm

print("Initializing summarization model... might take a few seconds.")
summarizer = pipeline("summarization", model="t5-small")
print("Model ready!\n")

def summarize_text(text):
    """
    Summarize given feedback into short and detailed versions.
    Handles errors gracefully if model fails.
    """
    try:
        short_sum = summarizer(text, max_length=60, min_length=20, do_sample=False)[0]["summary_text"]
        detailed_sum = summarizer(text, max_length=120, min_length=40, do_sample=False)[0]["summary_text"]
    except Exception as e:
        print(f"Error summarizing text: {e}")
        short_sum, detailed_sum = "", ""
    return short_sum, detailed_sum



input_path = "../data/cleaned_feedback.csv"
try:
    df = pd.read_csv(input_path)
except FileNotFoundError:
    print(f"File not found: {input_path}")
    exit()

print(f"Loaded {len(df)} records from {input_path}")


sample_df = df.head(100).copy()

short_list = []
detailed_list = []

print("\nStarting summarization process...\n")

for text in tqdm(sample_df["feedback_text"], desc="Summarizing", ncols=80):
    short, detailed = summarize_text(str(text))
    short_list.append(short)
    detailed_list.append(detailed)

sample_df["short_summary"] = short_list
sample_df["detailed_summary"] = detailed_list

output_file = "feedback_with_summaries_100.csv"
sample_df.to_csv(output_file, index=False)

print(f"\nDone! Summaries saved to '{output_file}'.")
print("Preview of first few results:")
print(sample_df[["feedback_text", "short_summary"]].head())
