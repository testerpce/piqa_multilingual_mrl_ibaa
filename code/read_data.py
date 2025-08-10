from utils import *


def get_args():
    parser = argparse.ArgumentParser(description="MLX LM Dataset and Model Loader")
    parser.add_argument("--dataset", type=str, default="baber/piqa", help="Name of the dataset to load")
    parser.add_argument("--model", type=str, default="mlx-community/Qwen3-30B-A3B-Instruct-2507-6bit", help="Name of the mlx_lm model to load")  
    parser.add_argument("--em_model", type=int, default='Qwen/Qwen3-Embedding-0.6B', help="Embedding model to help with classification and embedding")
    args = parser.parse_args()
    
    return args

def extract_data_details(dataset):
    pass


def main():
    args = get_args()

if __name__=="__main__":
    main()