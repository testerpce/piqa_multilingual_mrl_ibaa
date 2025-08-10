import datasets
import mlx_lm
import argparse
from typing import List
import torch
import torch.nn.functional as F
import json
from torch import Tensor
from transformers import AutoTokenizer, AutoModel

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import List

def load_hf_lm(name: str):
    """
    Loads a Hugging Face model and tokenizer, optimized for GPU usage.

    Args:
        name (str): The name of the model on the Hugging Face Hub.

    Returns:
        A tuple containing the loaded model and tokenizer.
    """
    # Check if a GPU is available and set the device accordingly.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # For very large models, you might need to load in 4-bit or 8-bit to fit on the GPU.
    # This can be enabled with a quantization config.
    # bnb_config = BitsAndBytesConfig(
    #     load_in_4bit=True,
    #     bnb_4bit_quant_type="nf4",
    #     bnb_4bit_compute_dtype=torch.bfloat16
    # )

    print(f"Loading model: {name}...")
    model = AutoModelForCausalLM.from_pretrained(
        name,
        torch_dtype=torch.bfloat16, # Use bfloat16 for better performance on modern GPUs
        device_map="auto", # Automatically distributes the model across available GPUs
        # quantization_config=bnb_config # Uncomment to use 4-bit quantization
    )
    
    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(name)

    # Set a padding token if one isn't already defined. This is crucial for batching.
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    return model, tokenizer


def call_hf_llm_batch(model, tokenizer, queries: List[str], max_new_tokens: int = 512) -> List[str]:
    """
    Generates responses for a batch of queries using a Hugging Face model.

    Args:
        model: The loaded Hugging Face model.
        tokenizer: The loaded Hugging Face tokenizer.
        queries (List[str]): A list of query strings to generate responses for.
        max_new_tokens (int): The maximum number of new tokens to generate for each query.

    Returns:
        A list of generated text responses, corresponding to the input queries.
    """
    # Format all queries into the standard chat format.
    # The input is a list of message lists.
    messages_batch = [[{"role": "user", "content": query}] for query in queries]
    
    # The tokenizer applies the template to each message list in the batch.
    # We don't add a generation prompt here, as the model.generate() call handles it.
    prompt_batch = tokenizer.apply_chat_template(
        messages_batch, tokenize=False, add_generation_prompt=True
    )

    # Tokenize the batch of prompts.
    inputs = tokenizer(
        prompt_batch, 
        return_tensors="pt", 
        padding=True, 
        truncation=True
    ).to(model.device)

    # Generate responses for the entire batch at once.
    # This is where the GPU's parallel processing power is used.
    outputs = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=True, # Use sampling for more creative/varied outputs
        temperature=0.6,
        top_p=0.9,
    )

    # Decode the generated tokens back into text.
    # We slice the output to only decode the newly generated tokens, not the input prompt.
    input_token_len = inputs["input_ids"].shape[1]
    generated_tokens = outputs[:, input_token_len:]
    
    generated_texts = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

    return generated_texts



def last_token_pool(last_hidden_states: Tensor,
                 attention_mask: Tensor) -> Tensor:
    left_padding = (attention_mask[:, -1].sum() == attention_mask.shape[0])
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths]


def get_detailed_instruct(task_description: str, query: str) -> str:
    return f'Instruct: {task_description}\nQuery:{query}'

def get_embedding_model(name: str):
    tokenizer = AutoTokenizer.from_pretrained(name, padding_side='left')
    model = AutoModel.from_pretrained(name)

    return model,tokenizer

def call_mlx_llm(model, tokenizer, query: str, max_length: int = 8192) -> str:
    messages = [{"role": "user", "content": query}]
    prompt = tokenizer.apply_chat_template(
        messages, add_generation_prompt=True
    )

    text = mlx_lm.generate(model, tokenizer, prompt=prompt, verbose=True)

    return text

# What is an action type for above actions?

def clean_json_response(response: str) -> dict:
    """Cleans the LLM response (handles markdown) and parses the JSON."""
    try:
        # Remove potential markdown formatting (e.g., ```json ... ```)
        response = response.strip()
        if response.startswith("```json"):
            response = response[7:]
        if response.endswith("```"):
            response = response[:-3]
        
        data = json.loads(response)
        if isinstance(data, dict):
            for key in data:
                if not isinstance(data[key], list):
                    # Ensure values are always lists for consistency
                    data[key] = [data[key]]
        return data
    except json.JSONDecodeError:
        print(f"Error parsing JSON response: {response}")
        return None

def get_embedding(input_texts: List[str], max_length: int, model, tokenizer):
    max_length = 8192

    # Tokenize the input texts
    batch_dict = tokenizer(
        input_texts,
        padding=True,
        truncation=True,
        max_length=max_length,
        return_tensors="pt",
    )
    batch_dict.to(model.device)
    outputs = model(**batch_dict)
    embeddings = last_token_pool(outputs.last_hidden_state, batch_dict['attention_mask'])

    return embeddings

def get_args():
    parser = argparse.ArgumentParser(description="MLX LM Dataset and Model Loader")
    parser.add_argument("--dataset", type=str, default="baber/piqa", help="Name of the dataset to load")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3-30B-A3B-Instruct-2507-6bit", help="Name of the mlx_lm model to load")  
    parser.add_argument("--em_model", type=int, default='Qwen/Qwen3-Embedding-0.6B', help="Embedding model to help with classification and embedding")
    args = parser.parse_args()
    
    return args

def load_mlx_lm(name: str):
    model, tokenizer = mlx_lm.load(name)

    return model, tokenizer



def load_dataset(name: str):
    dataset = datasets.load_dataset(name)

    return dataset

def main():
    args= get_args()
    dataset = load_dataset(args.dataset)
    model, tokenizer = load_mlx_lm(args.model)
    em_model, em_tokenizer = get_embedding_model(args.em_model)
    print(f"Loaded dataset: {dataset}")
    print(f"Loaded model: {model}")
    print(f"Loaded embedding model: {em_model}")




if __name__=="__main__":
    main()