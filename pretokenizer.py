"""
Pre-tokenize TinyStories train and validation datasets for faster training.
This script tokenizes the datasets once and saves them to disk for reuse.
"""
import argparse
import os
import time
from typing import List
import numpy as np
import torch
from datasets import load_dataset
from transformers import AutoTokenizer
from tqdm import tqdm

# uv run utils/pretokenizer.py --dict_size 10000 --tokenizer_path /shared_data0/hnwong/cache/tokenizer_tinystories_gpt2_10k --max_length 512


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-tokenize TinyStories datasets")
    
    # Model and tokenizer
    parser.add_argument("--model_id", type=str, default="roneneldan/TinyStories-33M", 
                       help="Hugging Face model repo id")
    parser.add_argument("--max_length", type=int, default=256, 
                       help="Maximum sequence length")
    parser.add_argument('--dict_size', type=int, default=50000, help='Dictionary size')
    # if there isnt any then load from huggingface, otherwise load from the path
    parser.add_argument('--tokenizer_path', type=str, default=None, help='Tokenizer path')
    
    # Dataset options
    parser.add_argument("--max_train_samples", type=int, default=None, 
                       help="Maximum number of training samples (None for all)")
    parser.add_argument("--max_val_samples", type=int, default=None, 
                       help="Maximum number of validation samples (None for all)")
    
    # Output options
    parser.add_argument("--output_dir", type=str, default="/shared_data0/hnwong/cache", 
                       help="Directory to save tokenized data")
    
    return parser.parse_args()


def tokenize_dataset(tokenizer: AutoTokenizer, dataset, max_length: int, 
                    max_samples: int = None, split_name: str = "train") -> List[torch.Tensor]:
    """Tokenize a dataset and return list of tokenized tensors"""
    print(f"\nTokenizing {split_name} dataset...")
    print(f"Max length: {max_length}")
    print(f"Max samples: {max_samples if max_samples else 'All'}")
    
    tokenized_data = []
    start_time = time.time()
    
    # Determine how many samples to process
    total_samples = len(dataset)
    if max_samples is not None:
        total_samples = min(total_samples, max_samples)
    
    print(f"Processing {total_samples:,} samples...")
    
    for i, item in enumerate(tqdm(dataset, desc=f"Tokenizing {split_name}", total=total_samples)):
        if max_samples is not None and i >= max_samples:
            break
            
        try:
            # Tokenize the text
            tokens = tokenizer(
                item['text'],
                max_length=max_length,
                truncation=True,
                padding='max_length',  # Pad to max_length
                return_tensors='pt'
            )['input_ids'].squeeze(0)  # Remove batch dimension

            #print(tokens.shape)
            
            tokenized_data.append(tokens)
            
        except Exception as e:
            print(f"Warning: Failed to tokenize sample {i}: {e}")
            continue
    
    elapsed_time = time.time() - start_time
    print(f"Tokenized {len(tokenized_data):,} samples in {elapsed_time:.2f} seconds")
    print(f"Average time per sample: {elapsed_time/len(tokenized_data)*1000:.2f}ms")
    
    return tokenized_data


def save_tokenized_data(tokenized_data: List[torch.Tensor], output_path: str, 
                       split_name: str, max_length: int) -> None:
    """Save tokenized data to disk as numpy array"""
    print(f"\nSaving {split_name} tokenized data to {output_path}...")
    
    # Create output directory
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Stack tensors into numpy array
    tokens_array = torch.stack(tokenized_data).numpy()
    
    # Save as numpy
    np.save(output_path, tokens_array)
    
    # Calculate file size
    file_size = os.path.getsize(output_path) / (1024**3)  # GB
    print(f"Saved {len(tokenized_data):,} samples to {output_path}")
    print(f"File size: {file_size:.2f} GB")
    print(f"Average size per sample: {file_size/len(tokenized_data)*1024:.2f} MB")


def load_tokenized_data(file_path: str) -> np.ndarray:
    """Load tokenized data from disk"""
    print(f"Loading tokenized data from {file_path}...")
    
    tokens = np.load(file_path)
    
    print(f"Loaded {len(tokens):,} samples with shape {tokens.shape}")
    
    return tokens


def main():
    args = get_args()
    
    print("=" * 80)
    print("TinyStories Dataset Pre-tokenization")
    print("=" * 80)
    print(f"Model: {args.model_id}")
    print(f"Max length: {args.max_length}")
    print(f"Output directory: {args.output_dir}")
    print("=" * 80)
    
    # Load tokenizer
    print("\nLoading tokenizer...")
    if args.tokenizer_path is None:
        print(f"Loading tokenizer from {args.model_id}...")
        tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)
    else:
        print(f"Loading own tokenizer from {args.tokenizer_path}...")
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    print(f"Tokenizer loaded: {tokenizer.__class__.__name__}")
    print(f"Vocab size: {tokenizer.vocab_size:,}")
    
    # Load datasets
    print("\nLoading TinyStories datasets...")
    train_dataset = load_dataset("roneneldan/TinyStories", split="train")
    val_dataset = load_dataset("roneneldan/TinyStories", split="validation")
    
    print(f"Train dataset: {len(train_dataset):,} samples")
    print(f"Validation dataset: {len(val_dataset):,} samples")
    
    # Define output paths
    train_output_path = os.path.join(args.output_dir, f"tinystories_train_maxlen_{args.max_length}_dict_{args.dict_size}.npy")
    val_output_path = os.path.join(args.output_dir, f"tinystories_val_maxlen_{args.max_length}_dict_{args.dict_size}.npy")
    
    # Check if files already exist
    if os.path.exists(train_output_path) and os.path.exists(val_output_path):
        print(f"\nTokenized files already exist!")
        print(f"Train: {train_output_path}")
        print(f"Val: {val_output_path}")
        
        response = input("Do you want to overwrite them? (y/N): ")
        if response.lower() != 'y':
            print("Skipping tokenization.")
            return
    
    # Tokenize training data
    train_tokenized = tokenize_dataset(
        tokenizer, train_dataset, args.max_length, 
        args.max_train_samples, "train"
    )
    
    # Tokenize validation data
    val_tokenized = tokenize_dataset(
        tokenizer, val_dataset, args.max_length, 
        args.max_val_samples, "validation"
    )
    
    # Save tokenized data
    save_tokenized_data(train_tokenized, train_output_path, "train", args.max_length)
    save_tokenized_data(val_tokenized, val_output_path, "validation", args.max_length)
    
    print("\n" + "=" * 80)
    print("Pre-tokenization completed successfully!")
    print("=" * 80)
    print(f"Train data: {train_output_path}")
    print(f"Val data: {val_output_path}")
    print("\nYou can now use these files in your training scripts for much faster data loading!")


if __name__ == "__main__":
    main()
