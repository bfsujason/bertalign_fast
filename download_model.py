#!/usr/bin/env python3

"""
Usage examples:
    python donwload_model.py
    python download_model.py --mirror hf-mirror.com
    
"""

import os
import argparse
from huggingface_hub import hf_hub_download

def main():
    parser = argparse.ArgumentParser(description="Download specific files from a Hugging Face model")
    parser.add_argument("--mirror", type=str, default=None,
                        help="Mirror site domain, e.g., hf-mirror.com. If not set, use official hub")
    parser.add_argument("--local-dir", type=str, default="./models/static-similarity-mrl-multilingual-v1",
                        help="Local directory to save the files")
    parser.add_argument("--repo-id", type=str,
                        default="sentence-transformers/static-similarity-mrl-multilingual-v1",
                        help="Hugging Face repository ID")
    parser.add_argument("--subfolder", type=str, default="0_StaticEmbedding",
                        help="Subfolder where the files are located")
    parser.add_argument("--files", nargs="+", default=["model.safetensors", "tokenizer.json"],
                        help="List of filenames to download (space separated)")
    parser.add_argument("--token", type=str, default=None,
                        help="Hugging Face authentication token (not required for public models)")
    args = parser.parse_args()

    # If a mirror is specified, set the HF_ENDPOINT environment variable
    if args.mirror:
        os.environ["HF_ENDPOINT"] = f"https://{args.mirror}"
        print(f"Using mirror site: {os.environ['HF_ENDPOINT']}")
    else:
        print("Using official Hugging Face Hub")

    # Create the local directory if it doesn't exist
    os.makedirs(args.local_dir, exist_ok=True)

    # Download each file individually
    for filename in args.files:
        print(f"\nStarting download: {args.subfolder}/{filename}")
        try:
            local_path = hf_hub_download(
                repo_id=args.repo_id,
                filename=filename,
                subfolder=args.subfolder,
                local_dir=args.local_dir,
                token=args.token
            )
            print(f"Download successful: {local_path}")
        except Exception as e:
            print(f"Download failed: {e}")

if __name__ == "__main__":
    main()
