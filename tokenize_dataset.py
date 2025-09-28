import os
import sys
import pathlib
import numpy as np
from itertools import chain
import argparse
from datasets import load_dataset
from transformers import AutoTokenizer

def main(args):

    data = load_dataset("roneneldan/TinyStories")
    print(f" Tokenizing with {args.tokenizer_name} tokenizer")

    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
    pathlib.Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    eos_token_id = tokenizer.eos_token_id

    tokenized_dataset = data.map(lambda example: tokenizer(
                    example["text"],
                    padding=False,
                    truncation=False,
                    add_special_tokens = False),
                    batched=True,
                    remove_columns=["text"])



    for split_name in ["train", "validation"]:
        print(f"\nConcatenating tokens for '{split_name}' split...") 
        # Concatenate all token ID lists, inserting the EOS token ID between them
        final_token_ids = []
        for example in tokenized_dataset[split_name]:
            final_token_ids.extend(example['input_ids'])
            final_token_ids.append(eos_token_id)
        output_path = os.path.join(args.output_dir, f"{split_name}.npy")
        print(f"Saving {len(final_token_ids)} tokens to {output_path}")
        np.save(output_path, np.array(final_token_ids, dtype=np.uint16))

    print("\nTokenization complete.")

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description="Tokenize TinyStories from HFHub")
    parser.add_argument(
        "--output_dir", 
        type=str, 
        default="./src/mintransformer/data/tokenized/tinystories", 
        help="Directory to save the output .npy files."
    )
    parser.add_argument(
        "--tokenizer_name", 
        type=str, 
        default="openai-community/gpt2", 
        help="Name of the Hugging Face tokenizer to use."
    )
    
    args = parser.parse_args()
    main(args)

