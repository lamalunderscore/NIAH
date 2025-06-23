"""Module that enables predictions in the NIAH benchmark."""

import glob
import json
import sys
from os import getenv
from pathlib import Path

import torch
import yaml
from safetensors.torch import save_file as save_tensor

from .utils import BackEnd, load_model


# If python does not find recurrent_gemma, add to correct directory to path:
sys.path.append(".")

CONF_FILE = "config.yaml"


def _save_attention_by_mode(
    tensors: dict[str, torch.Tensor],
    k: int,
    mode: str,
    attn_dir: Path,
    file_path: Path,
    min_needle: int | None = None,
):
    tokenizer, prompt_length, depth, rest = file_path.stem.split("_")
    if min_needle is not None:
        depth = min_needle
    save_path = str(attn_dir / f"k{k}_{mode}_{tokenizer}_{prompt_length}_{depth}.safetensors")
    save_tensor(tensors, save_path)
    print(f"✅ Saved attention weights to: {save_path}")


def run_predictions(config_file: str = CONF_FILE):  # noqa
    print(f"visible devices: {getenv('CUDA_VISIBLE_DEVICES')}")
    try:
        config_path = Path(__file__).resolve().parent / config_file
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
        parent_dir = Path(config["parent_dir"]).resolve()
        prompt_dir = parent_dir / config["prompt"]["save_dir"]
        save_dir = parent_dir / config["pred"]["save_dir"]

        model_type = config["pred"]["model_type"].lower()
        tokenizer_type = config["prompt"]["tokenizer"]["tokenizer_type"]
        model_path = config["pred"]["model_path"]
        batch_size = config["pred"]["batch_size"]

        k_indeces = config["pred"]["sparsification"]["k"]
        metric = config["pred"]["sparsification"]["metric"]
        do_sparse_prefill = config["pred"]["sparsification"]["sparse_prefill"]

        weight_manipulation = config["pred"]["weight_manipulation"]["manipulate"]
        if weight_manipulation:
            gen_mode = config["pred"]["weight_manipulation"]["gen_mode"]
            prefill_mode = config["pred"]["weight_manipulation"]["prefill_mode"]
            needle: str = config["prompt"]["needle"]

        save_attn = config["pred"]["save_attn"]["save"]
        if save_attn and batch_size != 1:
            print(
                "Warning: saving attention weights only works with batch_size=1, setting batch_size to 1."
            )
            batch_size = 1
        if save_attn:
            attn_dir: Path = parent_dir / config["pred"]["save_attn"]["attn_dir"]
            attn_get_mode = config["pred"]["save_attn"]["get_mode"]
            attn_dir.mkdir(parents=True, exist_ok=True)

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

        if save_attn:
            model.enable_attention_recording()

        # intervention
        for k in k_indeces:
            print(f"\n🔹 Processing k={k}")
            k_save_dir = save_dir / f"K{k}"
            if not k_save_dir.exists():
                k_save_dir.mkdir(parents=True, exist_ok=True)
            model.enable_head_sparsification(k=k, metric=metric, prefill=do_sparse_prefill)
            # Process prompts in batches
            batch_number = 0
            try:
                for i in range(0, len(all_prompts), batch_size):
                    batch_prompts = all_prompts[i : i + batch_size]
                    batch_filenames = filenames[i : i + batch_size]
                    number_batches = (len(all_prompts) + batch_size - 1) // batch_size
                    batch_number = i // batch_size + 1
                    print(f"\n🔹 Processing batch {batch_number}/{number_batches}")
                    print(f"🔍 Batch size: {len(batch_prompts)} prompts")
                    print(f"🔍 File names: {batch_filenames} prompts")

                    if weight_manipulation:
                        assert isinstance(needle, str)
                        needle_indices = model.find_needle(needle, batch_prompts[0])
                        if needle_indices is not None:
                            model.enable_weight_manipulation(
                                needle_indices,
                                gen_mode=gen_mode,
                                prefill_mode=prefill_mode,
                            )
                        else:
                            print(
                                "Warning: Needle not found in sequence, skipping weight manipulation."
                            )

                    with torch.no_grad():
                        out_data = model(batch_prompts)

                    if save_attn:
                        assert isinstance(attn_dir, Path)
                        file_path = Path(batch_filenames[0])
                        min_needle = None
                        if weight_manipulation:
                            if needle_indices is not None:
                                min_needle = min(needle_indices)
                        if attn_get_mode == "both":
                            attn_gen, attn_prefill = model.get_recorded_attention(attn_get_mode)
                            _save_attention_by_mode(
                                attn_gen,  # type:ignore
                                k,
                                "gen",
                                attn_dir,
                                file_path,
                                min_needle,
                            )
                            _save_attention_by_mode(
                                attn_prefill,  # type:ignore
                                k,
                                "prefill",
                                attn_dir,
                                file_path,
                                min_needle,
                            )
                        else:
                            _save_attention_by_mode(
                                model.get_recorded_attention(attn_get_mode),  # type:ignore
                                k,
                                attn_get_mode,
                                attn_dir,
                                file_path,
                                min_needle,
                            )

                    for j, filename in enumerate(batch_filenames):
                        out_string = out_data[j]

                        print(f"🔍 Output length: {len(out_string) if out_string else 0} chars")
                        if not out_string or len(out_string.strip()) == 0:
                            raise ValueError("Empty output")

                        basename = filename.name
                        newname = basename.replace(".json", ".txt").replace("_prompts", "")
                        output_save_path = k_save_dir / f"{newname}"
                        with open(output_save_path, "w") as f:
                            f.write(out_string)
                        print(f"✅ Saved prediction: {output_save_path}")

                    model.disable_weight_manipulation()

                    print(f"Memory before empty cache in GB: {backend.allocated_memory}")
                    print(f"Peak device memory in GB: {backend.peak_memory_reserved}")
                    backend.empty_cache()
                    backend.reset_peak()
            except RuntimeError as e:
                raise RuntimeError(f"🚨 Error processing batch {batch_number}!\n{str(e)}") from e
            model.disable_head_sparsification()
        model.disable_attention_recording()

        print("🎉 All predictions completed successfully!")

    except Exception as e:
        raise Exception(
            f"🚨 Fatal error in script: {e}\nError type: {type(e)}\nError details: {str(e)}"
        ) from e


__all__ = ("run_predictions",)
