# pred.py
import yaml
import os
import glob
import json
import torch
import sys
from pathlib import Path


import sentencepiece as spm
from recurrentgemma import torch as recurrentgemma

# If python does not find recurrent_gemma, add to correct directory to path:
sys.path.append(".")

CONF_FILE = "config.yaml"


def find_sequence(inputs, needle):
    print(needle)
    print(inputs)
    needle_len = needle.size(0)
    input_len = inputs.size(0)
    for i in range(input_len - needle_len + 1):
        if torch.equal(inputs[i : i + needle_len], needle):
            return [pos for pos in range(i, i + needle_len)]
    return None


if __name__ == "__main__":
    try:
        config_path = Path(__file__).resolve().parent / CONF_FILE
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        parent_dir = Path(config["parent_dir"])
        prompt_dir = parent_dir / config["prompt"]["save_dir"]
        save_dir = parent_dir / config["pred"]["save_dir"]

        tokenizer_type = config["prompt"]["tokenizer"]["tokenizer_type"]
        model_path = config["pred"]["model_path"]
        tokenizer_path = config["pred"]["tokenizer_path"]

        k_indeces = config["pred"]["sparsification"]["k"]
        metric = config["pred"]["sparsification"]["metric"]
        prefill = config["pred"]["sparsification"]["prefill"]

        needle_focus = config.get("needle_focus")  # not implemented
        needle_str = config.get("needle_str")  # not implemented
        needle_scaling = config.get("needle_scaling")  # not implemented

        print(f"🔹 Prompt directory (relative): {prompt_dir}")
        print(f"🔹 Prompt directory (absolute): {os.path.abspath(prompt_dir)}")
        print(f"🔹 Tokenizer provider: {tokenizer_type}")

        device = "cpu"
        backend = dict(
            allocated_memory=lambda: 0,
            empy_cache=lambda: 0,
        )
        if torch.cuda.is_available():
            backend = dict(
                allocated_memory=torch.cuda.memory_allocated,
                empy_cache=torch.cuda.empty_cache,
            )
            device = "cuda"
        elif torch.backends.mps.is_available():
            backend = dict(
                allocated_memory=torch.mps.current_allocated_memory,
                empty_cache=torch.mps.empty_cache,
            )
            device = "mps"

        print(f"Running on device '{device}'")

        # Load parameters
        params = torch.load(model_path)
        params = {
            k: v.to(device=device, dtype=torch.bfloat16)
            for k, v in params.items()
        }
        preset = recurrentgemma.Preset.RECURRENT_GEMMA_2B_V1
        model_config = recurrentgemma.GriffinConfig.from_torch_params(
            params, preset=preset
        )
        model = recurrentgemma.Griffin(
            model_config, device=device, dtype=torch.bfloat16
        )
        model.load_state_dict(params)
        vocab = spm.SentencePieceProcessor()
        vocab.Load(tokenizer_path)
        if needle_focus:
            needle_ids = vocab.encode(needle_str, out_type=int)

        sampler = recurrentgemma.Sampler(model=model, vocab=vocab)

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

            prompt = (
                prompts["content"][0]["content"]
                + "\n"
                + prompts["content"][1]["content"]
            )
            total_chars += len(prompt)
            all_prompts.append(prompt)
            filenames.append(filename)
            print(f"🔍 Prompt from {filename} length: {len(prompt)} chars")

        print(f"🔍 Total number of prompts: {len(all_prompts)}")
        print(
            f"🔍 Average prompt length: {total_chars / len(all_prompts):.2f} chars"
        )

        # Define batch size
        BATCH_SIZE = 1  # Start with small batch size
        # intervention
        for k in k_indeces:
            k_save_dir = f"{save_dir}_K{k}"
            if not os.path.exists(k_save_dir):
                os.makedirs(k_save_dir)
            model.enable_sparsification(k=k, metric=metric, prefill=prefill)
            # Process prompts in batches
            for i in range(0, len(all_prompts), BATCH_SIZE):
                batch_prompts = all_prompts[i : i + BATCH_SIZE]
                batch_filenames = filenames[i : i + BATCH_SIZE]

                print(
                    f"\n🔹 Processing batch {i//BATCH_SIZE + 1}/{(len(all_prompts) + BATCH_SIZE - 1)//BATCH_SIZE}"
                )
                print("🔍 Batch size: {len(batch_prompts)} prompts")
                print("🔍 Memory before batch:")
                print(
                    f"Allocated: {backend['allocated_memory']() / 1024**2:.2f}MB"
                )

                for i, (prompt, filename) in enumerate(
                    zip(batch_prompts, batch_filenames)
                ):
                    try:
                        if needle_focus:
                            model.enable_needle_focus(
                                find_sequence(
                                    torch.tensor(
                                        vocab.encode(prompt, out_type=int)
                                    ),
                                    needle_ids,
                                ),
                                needle_scaling,
                            )

                        out_data = sampler(
                            input_strings=[prompt], total_generation_steps=100
                        )

                        # Debug output data
                        print(f"🔍 Output data type: {type(out_data)}")
                        print(f"🔍 Output data attributes: {dir(out_data)}")
                        print(
                            f"🔍 Number of outputs: {len(out_data.text) if hasattr(out_data, 'text') else 'No text attribute'}"
                        )
                        out_string = out_data.text[0]
                        # Save results for this batch
                        print(
                            f"🔍 Output length: {len(out_string) if out_string else 0} chars"
                        )
                        if not out_string or len(out_string.strip()) == 0:
                            print(f"⚠️ Empty output for prompt from {filename}")
                            print(
                                f"🔍 Input prompt length: {len(prompt)} chars"
                            )
                            print(
                                f"🔍 First 100 chars of input prompt: {prompt[:100]}"
                            )
                            continue

                        basename = os.path.basename(filename)
                        newname = basename.replace(".json", ".txt").replace(
                            "_prompts", ""
                        )
                        save_path = f"{k_save_dir}/{newname}"
                        with open(save_path, "w") as f:
                            f.write(out_string)
                        print(f"✅ Saved prediction: {save_path}")
                        model.disable_needle_focus()

                        # Clear memory after each batch
                        backend["empty_cache"]()
                        print("🔍 CUDA memory after batch cleanup:")
                        print(
                            f"Allocated: {backend['allocated_memory']() / 1024**2:.2f}MB"
                        )
                    except RuntimeError as e:
                        print(
                            f"🚨 Error processing batch {i // BATCH_SIZE + 1}!"
                        )
                        print(f"Error details: {str(e)}")
                        raise e

        print("🎉 All predictions completed successfully!")

    except Exception as e:
        print(f"🚨 Fatal error in script: {e}")
        print(f"Error type: {type(e)}")
        print(f"Error details: {str(e)}")
        exit(1)
