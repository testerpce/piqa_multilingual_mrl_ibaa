import pandas as pd
import argparse
import json
from tqdm import tqdm
import os
from utils import * # Assuming this contains a BATCH-ENABLED call_mlx_llm_batch




# --- Script Configuration ---
classification_prompt_template = """
Your task is to classify the given text into a single, most appropriate label for each of the following categories.
You MUST choose exactly one label from the provided list for each category.
The output must be a single, valid JSON object.

Categories and Approved Labels:
{taxonomy_json}

Now, classify the following text:
text: {input}
Classification:
"""

# --- Functions ---
def get_args():
    parser = argparse.ArgumentParser(description="Batch Classification Workflow for RunPod")
    parser.add_argument("--dataset", type=str, default="baber/piqa", help="Hugging Face dataset to load")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3-30B-A3B-Instruct-2507-6bit", help="MLX language model for generation")
    parser.add_argument("--taxonomy_file", type=str, default="data/final_taxonomy.json", help="Path to the human-approved taxonomy JSON file")
    parser.add_argument("--output_file", type=str, default="data/piqa_classified.csv", help="Path to save the final classified CSV")
    parser.add_argument("--batch_size", type=int, default=64, help="Number of prompts to send to the LLM at once")
    return parser.parse_args()

def load_full_dataset(dataset_name: str) -> pd.DataFrame:
    """Loads the full dataset without sampling."""
    try:
        from datasets import load_dataset
        print(f"Loading full dataset: {dataset_name}")
        dataset_dict = load_dataset(dataset_name)
        train_data = dataset_dict['train'].to_pandas()
        validation_data = dataset_dict['validation'].to_pandas()
        return pd.concat([train_data, validation_data], ignore_index=True).reset_index(drop=True)
    except Exception as e:
        print(f"Could not load dataset: {e}. Returning dummy data.")
        return pd.DataFrame([{"goal": f"Full dataset goal {i}"} for i in range(2000)])

def main():
    args = get_args()

    # --- 1. Load Model, Taxonomy, and Data ---
    print("Loading model into GPU memory...")
    lm_model, lm_tokenizer = load_hf_lm(args.model)

    print(f"Loading taxonomy from {args.taxonomy_file}...")
    if not os.path.exists(args.taxonomy_file):
        raise FileNotFoundError(f"Taxonomy file not found! Please upload '{args.taxonomy_file}' to your RunPod instance.")
    with open(args.taxonomy_file, 'r') as f:
        final_taxonomy = json.load(f)
    
    taxonomy_json_str = json.dumps(final_taxonomy, indent=2)

    print("Loading full dataset...")
    full_df = load_full_dataset(args.dataset)

    # --- 2. Prepare All Prompts ---
    print("Preparing all prompts...")
    all_prompts = [
        classification_prompt_template.format(taxonomy_json=taxonomy_json_str, input=goal)
        for goal in full_df['goal']
    ]

    # --- 3. Process in Batches ---
    print(f"Starting batch classification with batch size {args.batch_size}...")
    all_classifications = []
    for i in tqdm(range(0, len(all_prompts), args.batch_size), desc="Classifying Batches"):
        batch_prompts = all_prompts[i:i + args.batch_size]
        
        # This is the key step: sending a whole batch to the GPU
        batch_responses_str = call_hf_llm_batch(lm_model, lm_tokenizer, batch_prompts)
        
        batch_classifications = [clean_json_response(resp) for resp in batch_responses_str]
        all_classifications.extend(batch_classifications)

    # --- 4. Merge and Save Results ---
    print("Merging classifications and saving final dataset...")
    classifications_df = pd.json_normalize(all_classifications).add_prefix('final_')
    
    final_df = full_df.join(classifications_df)

    # Ensure output directory exists
    output_dir = os.path.dirname(args.output_file)
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    final_df.to_csv(args.output_file, index=False)
    print(f"✅ Classified dataset saved to {args.output_file}")
    
    print("\n\nWorkflow complete. Download your classified dataset.")
    print("Final DataFrame head:")
    print(final_df.head())

if __name__ == '__main__':
    main()
