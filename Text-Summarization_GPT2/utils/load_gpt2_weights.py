"""
This script downloads the GPT-2 model weights and loads them into a dictionary.
"""

import os
import requests
import json
import numpy as np
import tensorflow as tf
from tqdm import tqdm

def download_and_load_gpt2(model_size="124M", models_dir="./model_weights/"):
    # Validate model size
    allowed_sizes = ("124M", "355M", "774M", "1558M")
    if model_size not in allowed_sizes:
        raise ValueError(f"Model size not in {allowed_sizes}")

    # Define paths
    model_dir = os.path.join(models_dir, model_size)
    base_url = "https://openaipublic.blob.core.windows.net/gpt-2/models"
    filenames = [
        "checkpoint", "encoder.json", "hparams.json",
        "model.ckpt.data-00000-of-00001", "model.ckpt.index",
        "model.ckpt.meta", "vocab.bpe"
    ]

    print(f"Downloading GPT-2 {model_size} model to {model_dir}")
    
    # Download files
    os.makedirs(model_dir, exist_ok=True)
    for filename in filenames:
        file_url = f"{base_url}/{model_size}/{filename}"
        file_path = os.path.join(model_dir, filename)
        download_file(file_url, file_path)

    print("Download completed. Loading model parameters...")

    # Load settings and params
    tf_ckpt_path = tf.train.latest_checkpoint(model_dir)
    settings = json.load(open(os.path.join(model_dir, "hparams.json")))
    params = load_gpt2_params_from_tf_ckpt(tf_ckpt_path, settings)

    print("Model loaded successfully!")
    print(f"Model configuration: {settings}")
    
    return settings, params

def download_file(url, destination):
    try:
        # Send a GET request to download the file, disabling SSL verification
        response = requests.get(url, stream=True, verify=False)
        response.raise_for_status()  # Raise an exception for bad status codes

        # Get the total file size from headers, defaulting to 0 if not present
        file_size = int(response.headers.get("content-length", 0))

        # Check if file exists and has the same size
        if os.path.exists(destination):
            file_size_local = os.path.getsize(destination)
            if file_size == file_size_local and file_size > 0:
                print(f"File already exists and is up-to-date: {destination}")
                return

        # Define the block size for reading the file
        block_size = 1024  # 1 Kilobyte

        # Initialize the progress bar with total file size
        progress_bar_description = url.split("/")[-1]  # Extract filename from URL
        with tqdm(total=file_size, unit="iB", unit_scale=True, desc=progress_bar_description) as progress_bar:
            # Open the destination file in binary write mode
            with open(destination, "wb") as file:
                # Iterate over the file data in chunks
                for chunk in response.iter_content(block_size):
                    if chunk:  # Filter out keep-alive chunks
                        progress_bar.update(len(chunk))  # Update progress bar
                        file.write(chunk)  # Write the chunk to the file

        print(f"Downloaded: {destination}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading the file: {e}")
        print(f"Please check the URL: {url}")
        raise

def load_gpt2_params_from_tf_ckpt(ckpt_path, settings):
    # Initialize parameters dictionary with empty blocks for each layer
    params = {"blocks": [{} for _ in range(settings["n_layer"])]}

    # Iterate over each variable in the checkpoint
    for name, _ in tf.train.list_variables(ckpt_path):
        # Load the variable and remove singleton dimensions
        variable_array = np.squeeze(tf.train.load_variable(ckpt_path, name))

        # Process the variable name to extract relevant parts
        variable_name_parts = name.split("/")[1:]  # Skip the 'model/' prefix

        # Identify the target dictionary for the variable
        target_dict = params
        if variable_name_parts[0].startswith("h"):
            layer_number = int(variable_name_parts[0][1:])
            target_dict = params["blocks"][layer_number]

        # Recursively access or create nested dictionaries
        for key in variable_name_parts[1:-1]:
            target_dict = target_dict.setdefault(key, {})

        # Assign the variable array to the last key
        last_key = variable_name_parts[-1]
        target_dict[last_key] = variable_array

    return params

if __name__ == "__main__":
    try:
        # Download and load GPT-2 model
        settings, params = download_and_load_gpt2(model_size="124M")
        
        # Print some basic info about the loaded model
        print("\nModel details:")
        print(f"- Vocabulary size: {settings.get('n_vocab', 'Unknown')}")
        print(f"- Number of layers: {settings.get('n_layer', 'Unknown')}")
        print(f"- Number of attention heads: {settings.get('n_head', 'Unknown')}")
        print(f"- Embedding dimension: {settings.get('n_embd', 'Unknown')}")
        print(f"- Context length: {settings.get('n_ctx', 'Unknown')}")
        
    except Exception as e:
        print(f"Error: {e}")
        print("Make sure you have the required dependencies installed:")
        print("pip install tensorflow requests tqdm numpy")