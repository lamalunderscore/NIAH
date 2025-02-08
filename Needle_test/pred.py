# pred.py
import yaml
import os
import glob
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

import inspect
import json
import torch
from typing import Optional, Union, Tuple

import traceback
import sys

import pathlib
import torch

import sentencepiece as spm
from recurrentgemma import torch as recurrentgemma

# import recurrentgemma

# If python does not find recurrent_gemma, add to correct directory to path:
sys.path.append(".")


def find_sequence(inputs, needle):
    needle_len = needle.size(0)
    input_len = inputs.size(0)
    for i in range(input_len - needle_len + 1):
        if torch.equal(inputs[i : i + needle_len], needle):
            return [pos for pos in range(i, i + needle_len)]
    return None


def pred(
    model_name,
    model,
    tokenizer,
    input_data,
    device,
    max_new_tokens=1024,
    temperature=0.1,
):
    try:
        print("🔹 Running prediction...")
        prompt = (
            input_data["content"][0]["content"]
            + "\n"
            + input_data["content"][1]["content"]
        )
        # # Extract prompt consistently
        # if isinstance(input_data, list) and len(input_data) >= 2:
        #     prompt = input_data[0]['content'] + '\n' + input_data[1]['content']
        # elif isinstance(input_data, dict) and 'needle' in input_data:
        #     prompt = input_data['needle']
        # else:
        #     print("🚨 Error: Unrecognized input format")
        #     return ""

        print(f"🔹 Using prompt: {prompt[:100]}...")

        # Basic tokenization - keep it simple
        inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(
            device
        )
        context_length = inputs.input_ids.shape[-1]

        # Use same generation call for all models
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            temperature=temperature,
        )[0]

        pred = tokenizer.decode(
            output[context_length:], skip_special_tokens=True
        )
        return pred.strip()

    except Exception as e:
        print(f"🚨 Error in prediction: {e}")
        print("🔍 Full traceback:")
        traceback.print_exc()  # This prints the full traceback
        return ""


def load_model_and_tokenizer(path, device):
    try:
        print(f"🔹 Loading model from {path}...")
        valid_path = path.lower()
        print(f"🔹 Checking model path: {valid_path}")

        if "longchat" in valid_path or "vicuna" in valid_path:
            from fastchat.model import load_model

            model, _ = load_model(
                path,
                device="cpu",
                num_gpus=0,
                load_8bit=False,
                cpu_offloading=False,
                debug=False,
            )
            model = model.to(device)
            model = model.bfloat16()
            tokenizer = AutoTokenizer.from_pretrained(
                path, trust_remote_code=True, use_fast=False
            )
        elif "mistral" in valid_path or "mixtral" in valid_path:
            tokenizer = AutoTokenizer.from_pretrained(
                path, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                path,
                use_flash_attention_2=True,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
            model.generation_config = GenerationConfig.from_pretrained(path)
        elif "recurrent_gemma" in valid_path:
            print(
                "🔹 Detected RecurrentGemma model, loading with custom logic..."
            )
            model = RecurrentGemmaForCausalLM.from_pretrained(path)
            model.to(device)
            tokenizer = AutoTokenizer.from_pretrained(
                path, trust_remote_code=True
            )
            print("✅ RecurrentGemma model loaded successfully!")
        else:
            tokenizer = AutoTokenizer.from_pretrained(
                path, trust_remote_code=True
            )
            model = AutoModelForCausalLM.from_pretrained(
                path,
                trust_remote_code=True,
                torch_dtype=torch.bfloat16,
                device_map="auto",
            )
        model = model.eval()
        print("✅ Model and tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"🚨 Error loading model: {e}")
        exit(1)


if __name__ == "__main__":
    try:
        print("🔹 Loading configuration...")
        with open(
            "/content/LongAlign/Needle_test/config-pred.yaml", "r"
        ) as file:
            config = yaml.load(file, Loader=yaml.FullLoader)

        model_provider = config["model"]["model_provider"]
        model_name = config["model"]["model_name"]
        prompt_dir = config["prompt_dir"]
        save_dir = config["save_dir"]
        k_indeces = config["sparsification"]["k"]
        sparse_metric = config["sparsification"]["metric"]
        sparse_prefill = config["sparsification"]["prefill"]
        needle_focus = config["needle"]["focus"]
        needle_scaling = config["needle"]["scaling"]
        needle_str = config["needle"]["needle"]

        print(f"🔹 Prompt directory (relative): {prompt_dir}")
        print(f"🔹 Prompt directory (absolute): {os.path.abspath(prompt_dir)}")
        print(f"🔹 Model provider: {model_provider}")

        # if not os.path.exists(save_dir):
        #     os.makedirs(save_dir)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Load parameters
        params = torch.load(
            "/root/.cache/kagglehub/models/google/recurrentgemma/PyTorch/2b/1/2b.pt"
        )
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
        vocab.Load(
            "/root/.cache/kagglehub/models/google/recurrentgemma/PyTorch/2b/1/tokenizer.model"
        )
        if needle_focus:
            needle_ids = torch.tensor(vocab.encode(needle_str, out_type=int))

        sampler = recurrentgemma.Sampler(model=model, vocab=vocab)

        pattern = f"{prompt_dir}/{model_provider}_*_prompts.json"
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
            model.enable_sparsification(
                k, metric=sparse_metric, prefill=sparse_prefill
            )
            # Process prompts in batches
            for i in range(0, len(all_prompts), BATCH_SIZE):
                batch_prompts = all_prompts[i : i + BATCH_SIZE]
                batch_filenames = filenames[i : i + BATCH_SIZE]

                print(
                    f"\n🔹 Processing batch {i//BATCH_SIZE + 1}/{(len(all_prompts) + BATCH_SIZE - 1)//BATCH_SIZE}"
                )
                print(f"🔹 Processing {batch_filenames}")

                for i, (prompt, filename) in enumerate(
                    zip(batch_prompts, batch_filenames)
                ):
                    print(f"🔍 Input size: {len(prompt)} characters")
                    print(
                        f"🔍 Input size: {len(vocab.encode(prompt, out_type=int))} tokens"
                    )
                    print("🔍 CUDA memory before batch:")
                    print(
                        f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f}MB"
                    )
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
                        in_strings = [prompt]
                        print(in_strings)
                        out_data = sampler(
                            input_strings=in_strings,
                            total_generation_steps=100,
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
                        torch.cuda.empty_cache()
                        print("🔍 CUDA memory after batch cleanup:")
                        print(
                            f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.2f}MB"
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
        traceback.print_exc()
        exit(1)
