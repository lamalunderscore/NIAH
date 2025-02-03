import yaml
import os
import glob
import json
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from llama_flash_attn_monkey_patch import replace_llama_attn_with_flash_attn

replace_llama_attn_with_flash_attn()

import inspect
import json
import torch
from typing import Optional, Union, Tuple

import traceback
import sys

# If python does not find recurrent_gemma, add to correct directory to path:
# sys.path.append("path/to/directory")

# Import the model

from recurrent_gemma.modeling_recurrent_gemma import RecurrentGemmaForCausalLM
from recurrent_gemma.configuration_recurrent_gemma import RecurrentGemmaConfig  # If needed



def pred(model_name, model, tokenizer, input_data, device, max_new_tokens=1024, temperature=0.1):
    try:
        print("🔹 Running prediction...")
        prompt = input_data['content'][0]['content'] + '\n' + input_data['content'][1]['content']
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
        inputs = tokenizer(prompt, truncation=False, return_tensors="pt").to(device)
        context_length = inputs.input_ids.shape[-1]

        # Use same generation call for all models
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            temperature=temperature,
        )[0]

        pred = tokenizer.decode(output[context_length:], skip_special_tokens=True)
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
            model, _ = load_model(path, device='cpu', num_gpus=0, load_8bit=False, cpu_offloading=False, debug=False)
            model = model.to(device)
            model = model.bfloat16()
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        elif "mistral" in valid_path or "mixtral" in valid_path:
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(path, use_flash_attention_2=True, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
            model.generation_config = GenerationConfig.from_pretrained(path)
        elif "recurrent_gemma" in valid_path:
            print("🔹 Detected RecurrentGemma model, loading with custom logic...")            
            model = RecurrentGemmaForCausalLM.from_pretrained(path)
            model.to(device)
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            print("✅ RecurrentGemma model loaded successfully!")
        else:
            tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True)
            model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, torch_dtype=torch.bfloat16, device_map="auto")
        model = model.eval()
        print("✅ Model and tokenizer loaded successfully!")
        return model, tokenizer
    except Exception as e:
        print(f"🚨 Error loading model: {e}")
        exit(1)

if __name__ == '__main__':
    try:
        print("🔹 Loading configuration...")
        with open('config-pred.yaml', 'r') as file:
            config = yaml.load(file, Loader=yaml.FullLoader)
        
        model_provider = config['model']['model_provider']
        model_name = config['model']['model_name']
        prompt_dir = config['prompt_dir']
        save_dir = config['save_dir']
        
        print(f"🔹 Prompt directory (relative): {prompt_dir}")
        print(f"🔹 Prompt directory (absolute): {os.path.abspath(prompt_dir)}")
        print(f"🔹 Model provider: {model_provider}")
        
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        model, tokenizer = load_model_and_tokenizer(model_name, device)
        
        pattern = f'{prompt_dir}/{model_provider}_*_prompts.json'
        print(f"🔹 Looking for prompt files matching pattern: {pattern}")
        print(f"🔹 Prompt directory exists: {os.path.exists(prompt_dir)}")
        if os.path.exists(prompt_dir):
            print(f"🔹 Contents of {prompt_dir}:")
            for f in os.listdir(prompt_dir):
                print(f"  - {f}")
                
        prompt_files = glob.glob(pattern)
        print(f"🔹 Found {len(prompt_files)} matching files")
        if not prompt_files:
            print("🚨 No prompt files found! Exiting...")
            exit(1)
        
        for filename in prompt_files:
            print(f"📂 Processing prompt file: {filename}")
            print(f"🔹 Loading prompts from: {os.path.abspath(filename)}")
            with open(filename, 'r') as f:
                prompts = json.load(f)
            
            if not prompts:
                print(f"⚠️ Empty prompt file: {filename}, skipping...")
                continue
            
            print("📨 Running model prediction...")
            result = pred(model_name.lower(), model, tokenizer, prompts, device)
            
            if result:
                basename = os.path.basename(filename)
                newname = basename.replace('.json', '.txt').replace('_prompts', '')
                save_path = f'{save_dir}/{newname}'
                with open(save_path, 'w') as f:
                    f.write(result)
                print(f"✅ Saved prediction: {save_path}")
            else:
                print("⚠️ No output generated.")
        
        print("🎉 Predictions completed successfully!")
    except Exception as e:
        print(f"🚨 Fatal error in script: {e}")
        exit(1)
