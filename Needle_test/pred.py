# pred.py
import glob
import json
import os
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict

import sentencepiece as spm
import torch
import yaml
from accelerate import (
    infer_auto_device_map,
    load_checkpoint_and_dispatch,
)
from accelerate.utils import get_balanced_memory
from recurrentgemma import torch as recurrentgemma


# If python does not find recurrent_gemma, add to correct directory to path:
sys.path.append(".")

CONF_FILE = "config.yaml"
BATCH_SIZE = 2


def find_sequence(inputs, needle):
    print(needle)
    print(inputs)
    needle_len = needle.size(0)
    input_len = inputs.size(0)
    for i in range(input_len - needle_len + 1):
        if torch.equal(inputs[i : i + needle_len], needle):
            return [pos for pos in range(i, i + needle_len)]
    return None


@dataclass
class BackEnd:
    allocated_memory: Callable[[], Dict[str, float]]
    empty_cache: Callable[[], None]


if __name__ == "__main__":
    print(f"visible devices: {os.getenv('CUDA_VISIBLE_DEVICES')}")
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
        backend = BackEnd(
            allocated_memory=lambda: 0,
            empty_cache=lambda: None,
        )
        if torch.cuda.is_available():
            backend = BackEnd(
                allocated_memory=lambda: {
                    str(i): torch.cuda.memory_allocated(torch.device(f"cuda:{i}")) / 1024**2
                    for i in range(torch.cuda.device_count())
                },
                empty_cache=torch.cuda.empty_cache,
            )
            device = "cuda"
        elif torch.backends.mps.is_available():
            backend = BackEnd(
                allocated_memory={"1": torch.mps.current_allocated_memory() / 1024**2},
                empty_cache=torch.mps.empty_cache,
            )
            device = "mps"

        print(f"Running on device '{device}'")

        # Load parameters

        params = torch.load(model_path)
        params = {k: v.to(device=device, dtype=torch.bfloat16) for k, v in params.items()}
        preset = (
            recurrentgemma.Preset.RECURRENT_GEMMA_2B_V1
            if "2b" in os.path.basename(model_path)
            else recurrentgemma.Preset.RECURRENT_GEMMA_9B_V1
        )
        model_config = recurrentgemma.GriffinConfig.from_torch_params(params, preset=preset)
        model = recurrentgemma.Griffin(model_config, device=device, dtype=torch.bfloat16)
        print(backend.allocated_memory())
        if torch.cuda.device_count() > 1:
            no_split_classes = [
                "ResidualBlock",
                "RecurrentBlock",
                "LocalAttentionBlock",
            ]
            print("Using accelerate for multi-GPU inference")
            balanced_mem = get_balanced_memory(
                model,
                no_split_module_classes=no_split_classes,
                low_zero=True,
            )
            print(f"balanced memory: {balanced_mem}")

            device_map = infer_auto_device_map(
                model,
                max_memory=balanced_mem,
                no_split_module_classes=no_split_classes,
            )
            model = load_checkpoint_and_dispatch(
                model, checkpoint=model_path, device_map=device_map
            )
        else:
            print("Using single-GPU setup")
            model.load_state_dict(params)
        assert model is not None, "Model cannot be none"
        model.eval()

        layers_per_device = defaultdict(int)
        for name, param in model.named_parameters():
            layers_per_device[param.device] += 1

        vocab = spm.SentencePieceProcessor()
        vocab.Load(tokenizer_path)
        if needle_focus:
            needle_ids = vocab.encode(needle_str, out_type=int)  # type: ignore

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

            prompt = prompts["content"]
            total_chars += len(prompt)
            all_prompts.append(prompt)
            filenames.append(filename)
            print(f"🔍 Prompt from {filename} length: {len(prompt)} chars")

        print(f"🔍 Total number of prompts: {len(all_prompts)}")
        print(f"🔍 Average prompt length: {total_chars / len(all_prompts):.2f} chars")

        # intervention
        for k in k_indeces:
            print(f"\n🔹 Processing k={k}")
            k_save_dir = save_dir / f"K{k}"
            if not os.path.exists(k_save_dir):
                os.makedirs(k_save_dir)
            model.enable_sparsification(k=k, metric=metric, prefill=prefill)
            # Process prompts in batches
            try:
                torch.cuda.memory._record_memory_history()
                for i in range(0, len(all_prompts), BATCH_SIZE):
                    batch_prompts = all_prompts[i : i + BATCH_SIZE]
                    batch_filenames = filenames[i : i + BATCH_SIZE]
                    number_batches = (len(all_prompts) + BATCH_SIZE - 1) // BATCH_SIZE
                    batch_number = i // BATCH_SIZE + 1
                    print(f"\n🔹 Processing batch {batch_number}/{number_batches}")
                    print(f"🔍 Batch size: {len(batch_prompts)} prompts")
                    print(f"🔍 File names: {batch_filenames} prompts")
                    with torch.no_grad():
                        out_data = sampler(
                            input_strings=batch_prompts,
                            total_generation_steps=40,
                        )

                    for j, filename in enumerate(batch_filenames):
                        out_string = out_data.text[j]

                        print(f"🔍 Output length: {len(out_string) if out_string else 0} chars")
                        if not out_string or len(out_string.strip()) == 0:
                            raise ValueError("Empty output")

                        basename = os.path.basename(filename)
                        newname = basename.replace(".json", ".txt").replace("_prompts", "")
                        save_path = k_save_dir / f"{newname}"
                        with open(save_path, "w") as f:
                            f.write(out_string)
                        print(f"✅ Saved prediction: {save_path}")
                        model.disable_needle_focus()
                    print(backend.allocated_memory())
                    backend.empty_cache()

            except RuntimeError as e:
                raise RuntimeError(f"🚨 Error processing batch {batch_number}!\n{str(e)}") from e

        print("🎉 All predictions completed successfully!")

    except Exception as e:
        raise Exception(
            f"🚨 Fatal error in script: {e}\nError type: {type(e)}\nError details: {str(e)}"
        ) from e
