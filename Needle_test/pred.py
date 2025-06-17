# pred.py
import glob
import json
import sys
from os import getenv
from pathlib import Path

import torch
import yaml
from utils import BackEnd, load_model


# If python does not find recurrent_gemma, add to correct directory to path:
sys.path.append(".")

CONF_FILE = "config.yaml"


if __name__ == "__main__":
    print(f"visible devices: {getenv('CUDA_VISIBLE_DEVICES')}")
    try:
        config_path = Path(__file__).resolve().parent / CONF_FILE
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        parent_dir = Path(config["parent_dir"])
        prompt_dir = parent_dir / config["prompt"]["save_dir"]
        save_dir = parent_dir / config["pred"]["save_dir"]

        model_type = config["pred"]["model_type"].lower()
        tokenizer_type = config["prompt"]["tokenizer"]["tokenizer_type"]
        model_path = config["pred"]["model_path"]
        tokenizer_path = config["pred"]["tokenizer_path"]
        batch_size = config["pred"]["batch_size"]

        k_indeces = config["pred"]["sparsification"]["k"]
        metric = config["pred"]["sparsification"]["metric"]
        prefill = config["pred"]["sparsification"]["prefill"]

        needle_focus = config.get("needle_focus")  # not implemented
        needle_str = config.get("needle_str")  # not implemented
        needle_scaling = config.get("needle_scaling")  # not implemented

        print(f"🔹 Prompt directory (relative): {prompt_dir}")
        print(f"🔹 Prompt directory (absolute): {prompt_dir.resolve()}")
        print(f"🔹 Tokenizer provider: {tokenizer_type}")

        backend = BackEnd()

        print(f"Running on device '{backend.device}'")

        # Load model
        model = load_model(model_type, model_path, device=backend.device)

        # set up prompts
        pattern = f"{prompt_dir}/{tokenizer_type}_*_prompts.json"
        print(f"🔹 Looking for prompt files matching pattern: {pattern}")
        prompt_files = glob.glob(pattern)
        print(f"🔹 Found {len(prompt_files)} matching files")
        if not prompt_files:
            print("🚨 No prompt files found! Exiting...")
            exit(1)

        # Collect all prompts and filenames
        all_prompts = []
        filenames = []
        total_chars = 0

        for filename in prompt_files:
            print(f"📂 Loading prompt file: {filename}")
            with open(filename, "r") as f:
                prompts = json.load(f)

            if not prompts:
                print(f"⚠️ Empty prompt file: {filename}, skipping...")
                continue

            prompt = prompts["content"]
            total_chars += len(prompt)
            all_prompts.append(prompt)
            filenames.append(Path(filename))
            print(f"🔍 Prompt from {filename} length: {len(prompt)} chars")

        print(f"🔍 Total number of prompts: {len(all_prompts)}")
        print(f"🔍 Average prompt length: {total_chars / len(all_prompts):.2f} chars")

        # intervention
        for k in k_indeces:
            print(f"\n🔹 Processing k={k}")
            k_save_dir = save_dir / f"K{k}"
            if not k_save_dir.exists():
                k_save_dir.mkdir(parents=True, exist_ok=True)
            model.model.enable_sparsification(k=k, metric=metric, prefill=prefill)
            # Process prompts in batches
            batch_number = 0
            try:
                torch.cuda.memory._record_memory_history()
                for i in range(0, len(all_prompts), batch_size):
                    batch_prompts = all_prompts[i : i + batch_size]
                    batch_filenames = filenames[i : i + batch_size]
                    number_batches = (len(all_prompts) + batch_size - 1) // batch_size
                    batch_number = i // batch_size + 1
                    print(f"\n🔹 Processing batch {batch_number}/{number_batches}")
                    print(f"🔍 Batch size: {len(batch_prompts)} prompts")
                    print(f"🔍 File names: {batch_filenames} prompts")
                    with torch.no_grad():
                        out_data = model(batch_prompts)

                    for j, filename in enumerate(batch_filenames):
                        out_string = out_data[j]

                        print(f"🔍 Output length: {len(out_string) if out_string else 0} chars")
                        if not out_string or len(out_string.strip()) == 0:
                            raise ValueError("Empty output")

                        basename = filename.name
                        newname = basename.replace(".json", ".txt").replace("_prompts", "")
                        save_path = k_save_dir / f"{newname}"
                        with open(save_path, "w") as f:
                            f.write(out_string)
                        print(f"✅ Saved prediction: {save_path}")
                    print(backend.allocated_memory)
                    backend.empty_cache()
            except RuntimeError as e:
                raise RuntimeError(f"🚨 Error processing batch {batch_number}!\n{str(e)}") from e
            model.model.disable_sparsification()

        print("🎉 All predictions completed successfully!")

    except Exception as e:
        raise Exception(f"🚨 Fatal error in script: {e}\nError type: {type(e)}\nError details: {str(e)}") from e
