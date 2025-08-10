import datasets
import mlx_lm
import argparse
from typing import List
import torch
import torch.nn.functional as F
import json
from torch import Tensor
from transformers import AutoTokenizer, AutoModel


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