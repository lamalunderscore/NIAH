# pred.py
import json
import sys
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file as save_tensor
from utils import BackEnd, load_model


# If python does not find recurrent_gemma, add to correct directory to path:
sys.path.append(".")

CONF_FILE = "config.yaml"


if __name__ == "__main__":
    try:
        config_path = Path(__file__).resolve().parent / CONF_FILE
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        parent_dir = Path(config["parent_dir"]).resolve()
        prompt_dir = parent_dir / config["prompt"]["save_dir"]
        save_dir = parent_dir / config["pred"]["save_dir"]
        save_dir.mkdir(parents=True, exist_ok=True)

        model_type = config["pred"]["model_type"].lower()
        tokenizer_type = config["prompt"]["tokenizer"]["tokenizer_type"]
        model_path = config["pred"]["model_path"]
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
        prompt_files = [
            str(prompt_dir / "Huggingface_250_100_prompts.json"),
            str(prompt_dir / "Huggingface_4096_0_prompts.json"),
        ]

        print(f"prompt files: {prompt_files}")

        # Collect all prompts and filenames
        all_prompts = []
        total_chars = 0

        for file_path in prompt_files:
            print(f"📂 Loading prompt file: {file_path}")
            with open(file_path, "r") as f:
                prompts = json.load(f)

            if not prompts:
                print(f"⚠️ Empty prompt file: {file_path}, skipping...")
                continue

            prompt = prompts["content"]
            total_chars += len(prompt)
            all_prompts.append(prompt)
            print(f"🔍 Prompt from {file_path} length: {len(prompt)} chars")

        print(f"🔍 Total number of prompts: {len(all_prompts)}")
        print(f"🔍 Average prompt length: {total_chars / len(all_prompts):.2f} chars")

        # intervention
        # Process prompts in batches
        batch_number = 0
        print(backend.allocated_memory)
        try:
            torch.cuda.memory._record_memory_history()
            for i in range(0, len(all_prompts), batch_size):
                model.enable_attention_recording()
                batch_prompts = all_prompts[i : i + batch_size]
                batch_filenames = prompt_files[i : i + batch_size]
                number_batches = (len(all_prompts) + batch_size - 1) // batch_size
                batch_number = i // batch_size + 1
                print(f"\n🔹 Processing batch {batch_number}/{number_batches}")
                print(f"🔍 Batch size: {len(batch_prompts)} prompts")
                print(f"🔍 File names: {batch_filenames} prompts")
                assert len(batch_filenames) == 1, "This only works with batch size=1"

                file_path = Path(batch_filenames[0])
                save_path = str(save_dir / file_path.with_suffix(".safetensors").name)

                with torch.no_grad():
                    out_data = model(batch_prompts)

                model.get_recorded_attention()

                save_tensor(model.get_recorded_attention(), save_path)  # type:ignore
                print(f"Saved to {save_path}")
                model.disable_attention_recording()

                backend.empty_cache()
        except RuntimeError as e:
            raise RuntimeError(f"🚨 Error processing batch {batch_number}!\n{str(e)}") from e

        print("🎉 All predictions completed successfully!")

    except Exception as e:
        raise Exception(f"🚨 Fatal error in script: {e}\nError type: {type(e)}\nError details: {str(e)}") from e
