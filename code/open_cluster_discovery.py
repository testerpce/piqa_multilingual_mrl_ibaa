import pandas as pd
import numpy as np
# from datasets import load_dataset # Uncomment when running
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import json
from tqdm import tqdm
from collections import defaultdict
import random
import os
from utils import *

DATA_DIR = "data" # Directory to save outputs

def get_args():
    parser = argparse.ArgumentParser(description="MLX LM Dataset and Model Loader")
    parser.add_argument("--dataset", type=str, default="baber/piqa", help="Name of the dataset to load")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3-30B-A3B-Instruct-2507-6bit", help="Name of the mlx_lm model to load")  
    parser.add_argument("--em_model", type=str, default='Qwen/Qwen3-Embedding-0.6B', help="Embedding model to help with classification and embedding")
    args = parser.parse_args()
    
    return args

# -----------------------------------------------------------------------------
# Configuration and Placeholders
# -----------------------------------------------------------------------------

# Define the high-level categories
CATEGORIES = ["Domain", "Core_Principle", "Key_Properties_Materials", "Action_Type"]


labelling_prompt = """
Your task is to analyze a given text and generate a structured JSON object containing meta-labels that describe it.

The JSON object must have the following four keys: "Domain", "Core_Principle", "Key_Properties_Materials", "Action_Type"


For each key, provide an array of relevant string labels.

Do not provide more than three labels for any given key.

The output must be a single, valid JSON object.

Label Definitions:

Domain: The general field or area the text belongs to (e.g., "Cooking", "Social Media").

Core_Principle: The fundamental concepts, theories, or ideas involved (e.g., "Heat Transfer", "Digital Navigation").

Key_Properties_Materials: The primary objects, substances, or components and their key characteristics (e.g., "Water (Liquid)", "Software Application").

Action_Type: The main actions or verbs described in the text (e.g., "Boiling", "Scrolling").

Example 1:

text: "How to boil eggs"

Meta-labels:

{{
  "Domain": ["Kitchen", "Cooking"],
  "Core_Principle": ["Heat Transfer", "Thermodynamics"],
  "Key_Properties_Materials": ["Water (Liquid)", "Egg (Fragile)"],
  "Action_Type": ["Heating", "Boiling"]
}}

Example 2:

text: "How to check your Facebook feed"

Meta-labels:

{{
  "Domain": ["Digital", "Social Media"],
  "Core_Principle": ["Digital Navigation", "UI Design"],
  "Key_Properties_Materials": ["Software Application", "Interface"],
  "Action_Type": ["Clicking", "Scrolling", "Browsing"]
}}

Now, generate the meta-labels for the following text:

text: {input}

Meta-labels:
"""

# This is the prompt for the final, constrained classification task
classification_prompt = """
Your task is to classify the given text into a single, most appropriate label for each of the following categories.
You MUST choose exactly one label from the provided list for each category.

The output must be a single, valid JSON object.

Categories and Approved Labels:
{taxonomy_json}

Now, classify the following text:
text: {input}
Classification:
"""

# -----------------------------------------------------------------------------
# Data Loading
# -----------------------------------------------------------------------------

def load_and_sample_piqa(sample_size=1000) -> pd.DataFrame:
    """Loads the PIQA dataset, combines splits, and takes a random sample."""
    try:
        dataset_dict = load_dataset("baber/piqa")
        train_data = dataset_dict['train'].to_pandas()
        validation_data = dataset_dict['validation'].to_pandas()

        combined_df = pd.concat([train_data, validation_data], ignore_index=True)
        if sample_size > len(combined_df):
            return combined_df
        return combined_df.sample(n=sample_size, random_state=42), combined_df 
        
    except Exception as e:
        print(f"Could not load dataset: {e}. Returning dummy data.")
        # Dummy Data if loading fails
        dummy_data = [
            {"goal": "To boil eggs, you should"},
            {"goal": "How to check your Facebook feed"},
            {"goal": "Convert unused paint to solid waste."},
            {"goal": "To make a stack of books more stable"}
        ]
        # Expand dummy data to simulate the sample size
        for i in range(sample_size - len(dummy_data)):
            dummy_data.append({"goal": f"Generic goal {i} about {random.choice(['fixing', 'making', 'using'])} things."})
        return pd.DataFrame(dummy_data), pd.DataFrame(dummy_data)

# -----------------------------------------------------------------------------
# Phase 1: Open Concept Generation
# -----------------------------------------------------------------------------

def extract_open_concepts(model, tokenizer, goal: str) -> dict:
    """
    Uses an LLM to extract descriptive concepts for a goal based on redefined categories.
    """
    # prompt = f"""
    # Analyze the following goal. Identify the fundamental concepts required to understand it. 
    
    # Categorize these concepts into the following four groups. Propose concise labels for each.

    # 1. Domain: The general area of activity (e.g., Kitchen, Electronics, Chemistry, Physics).
    # 2. Core_Principle: The underlying knowledge or mechanism required (e.g., Heat Transfer, Digital Navigation, Gravity, Safety Protocol).
    # 3. Key_Properties_Materials: Relevant characteristics of the objects involved (e.g., Fragile, Liquid/Viscous, Digital Interface).
    # 4. Action_Type: The primary action being performed (e.g., Heating, Mixing, Disposing, Clicking).

    # Goal: {goal}

    # Provide the output ONLY as a JSON object with the keys: {json.dumps(CATEGORIES)}. 
    # Each key should contain a list of relevant proposed labels.
    # """
    prompt = labelling_prompt.format(input=goal)
    response = call_mlx_llm(model, tokenizer, prompt)
    return clean_json_response(response)

def aggregate_concepts(df: pd.DataFrame) -> dict:
    """Aggregates the extracted concepts from the DataFrame."""
    aggregated = defaultdict(list)

    for index, row in df.iterrows():
        concepts = row['extracted_concepts']
        if concepts:
            for category in CATEGORIES:
                if category in concepts:
                    # Normalize case and spacing
                    normalized_labels = [label.strip().lower() for label in concepts[category]]
                    aggregated[category].extend(normalized_labels)
    
    # Return unique, sorted lists for clustering
    return {k: sorted(list(set(v))) for k, v in aggregated.items()}

# -----------------------------------------------------------------------------
# Phase 2: Embedding and Clustering (Consolidation)
# -----------------------------------------------------------------------------

def cluster_labels(model, labels: list, n_clusters_ratio=0.1):
    """
    Embeds labels and clusters them using K-Means to organize the taxonomy.
    """
    if not labels:
        return []

    # Determine the number of clusters. This ratio needs tuning. 
    # Start by aiming for 1 cluster per 10 unique labels.
    n_clusters = max(1, int(len(labels) * n_clusters_ratio))
    
    print(f"  Embedding {len(labels)} unique labels...")
    # In a real environment, load the model outside this function for efficiency
    #model = SentenceTransformer('all-mpnet-base-v2')
    embeddings = model.encode(labels, show_progress_bar=True)
    
    # DUMMY embeddings for execution without the model
    # embeddings = np.random.rand(len(labels), 768)
    # # Simulate some similarity for demonstration
    # if len(labels) > 5:
    #     embeddings[1] = embeddings[0] + np.random.normal(0, 0.05, 768)
    #     embeddings[5] = embeddings[4] + np.random.normal(0, 0.05, 768)


    # Normalize embeddings (important for K-Means)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    print(f"  Starting K-Means clustering (N_Clusters={n_clusters})...")
    
    # Handle case where n_clusters might be >= number of samples after randomization
    if n_clusters >= len(labels):
        n_clusters = len(labels)

    # Using 'auto' for n_init as 'warn' is the default and 'auto' is generally better
    clustering_model = KMeans(n_clusters=n_clusters, n_init='auto', random_state=42)
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_
    
    # Organize results
    clustered_sentences = [[] for i in range(n_clusters)]
    for sentence_id, cluster_id in enumerate(cluster_assignment):
        clustered_sentences[cluster_id].append(labels[sentence_id])
        
    return clustered_sentences


# -----------------------------------------------------------------------------
# Phase 3: Human-in-the-Loop Taxonomy Definition (NEW)
# -----------------------------------------------------------------------------

def define_taxonomy_interactively(taxonomy_clusters: dict) -> dict:
    """
    Guides a user through naming the discovered clusters to create a final,
    human-approved taxonomy.
    """
    print("\n--- Phase 3: Interactive Taxonomy Definition ---")
    print("You will be shown clusters of similar labels. Provide a canonical name for each.")
    print("For example, for ['heating', 'warming', 'boiling'], you might enter 'Heat Application'.")
    print("Press ENTER to skip a cluster if it doesn't make sense.\n")

    final_taxonomy = defaultdict(list)

    for category, clusters in taxonomy_clusters.items():
        print(f"\n===== Defining Category: {category.upper()} =====")
        for i, cluster in enumerate(clusters):
            if not cluster: continue
            
            print(f"\nCluster {i+1} (Size: {len(cluster)}): {cluster}")
            user_input = input("  Enter canonical name (or press ENTER to skip): ")
            
            if user_input.strip():
                final_taxonomy[category].append(user_input.strip())
                print(f"  > Saved '{user_input.strip()}'")
    
    # Ensure all categories are present in the final dict
    for category in CATEGORIES:
        if category not in final_taxonomy:
            final_taxonomy[category] = []
            
    print("\n--- Taxonomy Definition Complete ---")
    return dict(final_taxonomy)

# -----------------------------------------------------------------------------
# Phase 4: Apply Final Taxonomy to Dataset (NEW)
# -----------------------------------------------------------------------------

def _classify_single_goal(goal: str, taxonomy_json: str, model, tokenizer) -> dict:
    """Helper function to classify one goal using the final taxonomy."""
    prompt = classification_prompt.format(taxonomy_json=taxonomy_json, input=goal)
    response = call_mlx_llm(model, tokenizer, prompt)
    return clean_json_response(response)


def apply_final_taxonomy(df: pd.DataFrame, final_taxonomy: dict, model, tokenizer) -> pd.DataFrame:
    """Uses an LLM to classify each goal according to the final taxonomy."""
    print("\n--- Phase 4: Applying Final Taxonomy to Dataset ---")
    taxonomy_json_str = json.dumps(final_taxonomy, indent=2)
    
    tqdm.pandas(desc="Classifying Goals")
    classifications = df['goal'].progress_apply(
        lambda g: _classify_single_goal(g, taxonomy_json_str, model, tokenizer)
    )
    
    classifications_df = pd.json_normalize(classifications).add_prefix('final_')
    return df.join(classifications_df)




# -----------------------------------------------------------------------------
# Main Execution Flow (Discovery & Consolidation)
# -----------------------------------------------------------------------------

def main():
    args = get_args()
    # --- Configuration ---
    SAMPLE_SIZE = 200  # Recommended: 1000-2000
    CLUSTER_RATIO = 0.1 # Tune this: Higher ratio = more granular taxonomy

    # --- 1. Load and Sample Data ---
    print("Loading and sampling PIQA data...")
    sample_df, full_df = load_and_sample_piqa(SAMPLE_SIZE)
    lm_model, lm_tokenizer = load_mlx_lm(args.model)
    # --- 2. Phase 1: Extract Open Concepts ---
    print(f"Phase 1: Extracting concepts from {len(sample_df)} goals using ...")
    tqdm.pandas(desc="Extracting Concepts")
    
    # Apply the extraction
    sample_df['extracted_concepts'] = sample_df['goal'].progress_apply(
        lambda current_goal: extract_open_concepts(
            model=lm_model, 
            tokenizer=lm_tokenizer, 
            goal=current_goal
        )
    )
    # Aggregate all concepts
    all_concepts = aggregate_concepts(sample_df)

    # --- 3. Phase 2: Cluster Concepts (Consolidation) ---
    print("Phase 2: Clustering concepts...")
    print("Loading embedding model (e.g., all-mpnet-base-v2)...")
    embedding_model = SentenceTransformer('Qwen/Qwen3-Embedding-0.6B')# Initialize the embedding model here if running for real
    
    taxonomy_clusters = {}
    for category, labels in all_concepts.items():
        print(f"\nProcessing Category: {category}")
        taxonomy_clusters[category] = cluster_labels(embedding_model, labels, n_clusters_ratio=CLUSTER_RATIO)

    # --- 3. Human-in-the-Loop Taxonomy Definition ---
    final_taxonomy = define_taxonomy_interactively(taxonomy_clusters)
    
    # Save the human-approved taxonomy
    taxonomy_filepath = os.path.join(DATA_DIR, 'final_taxonomy.json')
    with open(taxonomy_filepath, 'w') as f:
        json.dump(final_taxonomy, f, indent=4)
    print(f"\n✅ Final taxonomy saved to {taxonomy_filepath}")

    # --- 4. Apply Final Taxonomy to Dataset ---
    # classified_df = apply_final_taxonomy(full_df, final_taxonomy, lm_model, lm_tokenizer)

    # # Save the final classified dataset
    # dataset_filepath = os.path.join(DATA_DIR, 'piqa_classified.csv')
    # classified_df.to_csv(dataset_filepath, index=False)
    # print(f"✅ Classified dataset saved to {dataset_filepath}")
    
    # print("\n\nWorkflow complete. Check the 'data' folder for your taxonomy and classified dataset.")
    # print("Final DataFrame head:")
    # print(classified_df.head())

if __name__ == '__main__':
    main()
    