from dotenv import load_dotenv
import os
import tiktoken
import glob
import json
import yaml
from anthropic import Anthropic
import numpy as np
import asyncio
from asyncio import Semaphore
from transformers import AutoTokenizer
from pathlib import Path

CONF_FILE = "config-prompt.py"  # name of the config file

load_dotenv()


class Prompter:
    def __init__(
        self,
        needle="\nThe best thing to do in San Francisco is eat a sandwich and sit in Dolores Park on a sunny day.\n",
        haystack_dir="PaulGrahamEssays",
        retrieval_question="What is the best thing to do in San Francisco?",
        context_lengths_min=1000,
        context_lengths_max=200000,
        context_lengths_num_intervals=35,
        context_lengths=None,
        document_depth_percent_min=0,
        document_depth_percent_max=100,
        document_depth_percent_intervals=35,
        document_depth_percents=None,
        document_depth_percent_interval_type="linear",
        tokenizer_type="OpenAI",
        model_name="gpt-4-1106-preview",
        num_concurrent_requests=1,
        final_context_length_buffer=200,
        save_dir="prompts",
        print_ongoing_status=True,
    ):

        print("🚀 Initializing Prompter...")
        if not all([needle, haystack_dir, retrieval_question]):
            raise ValueError(
                "Needle, haystack_dir, and retrieval_question must all be provided."
            )

        self.needle = needle
        self.haystack_dir = haystack_dir
        self.retrieval_question = retrieval_question
        self.num_concurrent_requests = num_concurrent_requests
        self.final_context_length_buffer = final_context_length_buffer
        self.print_ongoing_status = print_ongoing_status
        self.tokenizer_type = tokenizer_type
        self.model_name = model_name
        self.testing_results = []
        self.save_dir = save_dir

        print(f"📁 Haystack directory: {self.haystack_dir}")
        print(f"💾 Save directory: {self.save_dir}")

        self._setup_context_lengths(
            context_lengths,
            context_lengths_min,
            context_lengths_max,
            context_lengths_num_intervals,
        )
        self._setup_document_depth_percents(
            document_depth_percents,
            document_depth_percent_min,
            document_depth_percent_max,
            document_depth_percent_intervals,
            document_depth_percent_interval_type,
        )
        self._initialize_tokenizer()
        print("✅ Initialization complete\n")

    def _setup_context_lengths(
        self, context_lengths, min_len, max_len, num_intervals
    ):
        print("⚙️ Setting up context lengths...")
        if context_lengths is None:
            if any(x is None for x in [min_len, max_len, num_intervals]):
                raise ValueError(
                    "Either provide context_lengths list or all required parameters"
                )
            self.context_lengths = np.round(
                np.linspace(min_len, max_len, num=num_intervals, endpoint=True)
            ).astype(int)
        else:
            self.context_lengths = np.array(context_lengths)
        print(f"✅ Context lengths set: {self.context_lengths}")

    def _setup_document_depth_percents(
        self,
        depth_percents,
        min_percent,
        max_percent,
        intervals,
        interval_type,
    ):
        print("⚙️ Setting up document depth percentages...")
        if depth_percents is None:
            if interval_type == "linear":
                self.document_depth_percents = np.round(
                    np.linspace(
                        min_percent, max_percent, num=intervals, endpoint=True
                    )
                ).astype(int)
            elif interval_type == "sigmoid":
                self.document_depth_percents = [
                    self.logistic(x)
                    for x in np.linspace(min_percent, max_percent, intervals)
                ]
            else:
                raise ValueError("Invalid interval_type")
        else:
            self.document_depth_percents = np.array(depth_percents)
        print(f"✅ Depth percentages set: {self.document_depth_percents}")

    def _initialize_tokenizer(self):
        print(f"🔄 Initializing {self.tokenizer_type} tokenizer...")
        try:
            if self.tokenizer_type == "OpenAI":
                self.enc = tiktoken.encoding_for_model(self.model_name)
            elif self.tokenizer_type == "Anthropic":
                self.enc = Anthropic().get_tokenizer()
            elif self.tokenizer_type == "Huggingface":
                self._initialize_huggingface_tokenizer()
            else:
                raise ValueError(
                    f"Unsupported tokenizer_type: {self.tokenizer_type}"
                )
            print("✅ Tokenizer initialized successfully")
        except Exception as e:
            print(f"❌ Tokenizer initialization failed: {str(e)}")
            raise

    def _initialize_huggingface_tokenizer(self):
        print("🤗 Initializing HuggingFace tokenizer...")
        is_local = os.path.exists(self.model_name)
        if is_local:
            print(f"📂 Loading local tokenizer from {self.model_name}")
        else:
            print(
                f"🌐 Loading tokenizer from HuggingFace Hub: {self.model_name}"
            )

        try:
            if "gemma" in self.model_name.lower():
                self.enc = AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=True,
                    local_files_only=is_local,
                    trust_remote_code=True,
                )
            else:
                self.enc = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    use_fast=False,
                    local_files_only=is_local,
                )
            print("✅ HuggingFace tokenizer initialized successfully")
        except Exception as e:
            print(f"❌ HuggingFace tokenizer initialization failed: {str(e)}")
            raise

    @staticmethod
    def logistic(x):
        return int(100 / (1 + np.exp(-0.1 * (x - 50))))

    def encode_text_to_tokens(self, text):
        try:
            if self.tokenizer_type == "OpenAI":
                return self.enc.encode(text)
            elif self.tokenizer_type == "Anthropic":
                return self.enc.encode(text).ids
            elif self.tokenizer_type == "Huggingface":
                return (
                    self.enc(
                        text,
                        truncation=False,
                        return_tensors="pt",
                        add_special_tokens=False,
                    )
                    .input_ids.view(-1)
                    .tolist()
                )
        except Exception as e:
            print(f"❌ Token encoding failed: {str(e)}")
            raise

    def decode_tokens(self, tokens, context_length=None):
        try:
            if self.tokenizer_type == "OpenAI":
                return self.enc.decode(tokens[:context_length])
            elif self.tokenizer_type == "Anthropic":
                return self.enc.decode(tokens[:context_length])
            elif self.tokenizer_type == "Huggingface":
                return self.enc.decode(
                    tokens[:context_length], skip_special_tokens=True
                )
        except Exception as e:
            print(f"❌ Token decoding failed: {str(e)}")
            raise

    def insert_needle(self, context, depth_percent, context_length):
        print(f"🎯 Inserting needle at {depth_percent}% depth...")
        try:
            tokens_needle = self.encode_text_to_tokens(self.needle)
            tokens_context = self.encode_text_to_tokens(context)

            context_length -= self.final_context_length_buffer
            print(
                f"📊 Context tokens: {len(tokens_context)}, Needle tokens: {len(tokens_needle)}"
            )

            if len(tokens_context) + len(tokens_needle) > context_length:
                tokens_context = tokens_context[
                    : context_length - len(tokens_needle)
                ]
                print("✂️ Trimmed context to fit needle")

            if depth_percent == 100:
                tokens_new_context = tokens_context + tokens_needle
                print("📌 Needle inserted at end")
            else:
                insertion_point = int(
                    len(tokens_context) * (depth_percent / 100)
                )
                tokens_new_context = tokens_context[:insertion_point]

                period_tokens = self.encode_text_to_tokens(".")
                while (
                    tokens_new_context
                    and tokens_new_context[-1] not in period_tokens
                ):
                    insertion_point -= 1
                    tokens_new_context = tokens_context[:insertion_point]

                tokens_new_context += (
                    tokens_needle + tokens_context[insertion_point:]
                )
                print(f"📌 Needle inserted at position {insertion_point}")

            result = self.decode_tokens(tokens_new_context)
            print("✅ Needle insertion complete")
            return result
        except Exception as e:
            print(f"❌ Needle insertion failed: {str(e)}")
            raise

    def get_context_length_in_tokens(self, context):
        try:
            if self.tokenizer_type == "OpenAI":
                return len(self.enc.encode(context))
            elif self.tokenizer_type == "Anthropic":
                return len(self.enc.encode(context).ids)
            elif self.tokenizer_type == "Huggingface":
                return self.enc(
                    context, truncation=False, return_tensors="pt"
                ).input_ids.shape[-1]
        except Exception as e:
            print(f"❌ Context length calculation failed: {str(e)}")
            raise

    def read_context_files(self):
        print(f"📚 Reading files from {self.haystack_dir}")
        context = ""
        files_read = 0
        max_context_length = max(self.context_lengths)

        try:
            file_list = glob.glob(f"{self.haystack_dir}/*.txt")
            if not file_list:
                raise FileNotFoundError(
                    f"No .txt files found in {self.haystack_dir}"
                )

            print(f"📁 Found {len(file_list)} files")

            while (
                self.get_context_length_in_tokens(context) < max_context_length
            ):
                for file in file_list:
                    print(f"📄 Reading: {os.path.basename(file)}")
                    with open(file, "r") as f:
                        context += f.read()
                    files_read += 1

            print(f"✅ Read {files_read} files, total chars: {len(context)}")
            return context
        except Exception as e:
            print(f"❌ Error reading files: {str(e)}")
            raise

    def encode_and_trim(self, context, context_length):
        print(f"✂️ Trimming context to {context_length} tokens...")
        try:
            tokens = self.encode_text_to_tokens(context)
            if len(tokens) > context_length:
                context = self.decode_tokens(tokens, context_length)
                print(
                    f"✅ Trimmed from {len(tokens)} to {context_length} tokens"
                )
            return context
        except Exception as e:
            print(f"❌ Trimming failed: {str(e)}")
            raise

    async def bound_evaluate_and_log(self, sem, *args):
        async with sem:
            await self.evaluate_and_log(*args)

    async def evaluate_and_log(self, context_length, depth_percent):
        print(
            f"\n📋 Processing: length={context_length}, depth={depth_percent}%"
        )
        try:
            context = await self.generate_context(
                context_length, depth_percent
            )
            print(f"✅ Context generated: {len(context)} chars")

            prompt = {
                "model": self.model_name,
                "tokenizer": self.tokenizer_type,
                "context_length": int(context_length),
                "depth_percent": int(depth_percent),
                "needle": self.needle,
                "retrieval_question": self.retrieval_question,
                "content": [
                    {
                        "role": "system",
                        "content": "You are a helpful assistant.",
                    },
                    {
                        "role": "user",
                        "content": f"{context}\n\n{self.retrieval_question}",
                    },
                ],
            }

            filename = f"{self.tokenizer_type}_{context_length}_{depth_percent}_prompts.json"
            save_path = os.path.join(self.save_dir, filename)

            print(f"💾 Saving to: {save_path}")
            with open(save_path, "w") as f:
                json.dump(prompt, f, indent=2)
            print("✅ Save complete\n")

        except Exception as e:
            print(f"❌ Evaluation failed: {str(e)}")
            raise

    async def generate_context(self, context_length, depth_percent):
        print("🔄 Generating context...")
        try:
            context = self.read_context_files()
            context = self.encode_and_trim(context, context_length)
            result = self.insert_needle(context, depth_percent, context_length)
            print("✅ Context generation complete")
            return result
        except Exception as e:
            print(f"❌ Context generation failed: {str(e)}")
            raise

    async def run_test(self):
        print("🏃 Starting test run...")
        sem = Semaphore(self.num_concurrent_requests)
        tasks = []

        for context_length in self.context_lengths:
            for depth_percent in self.document_depth_percents:
                task = self.bound_evaluate_and_log(
                    sem, context_length, depth_percent
                )
                tasks.append(task)

        print(f"📊 Created {len(tasks)} tasks")
        await asyncio.gather(*tasks)
        print("✅ Test run complete")

    def start_test(self):
        print("\n🚀 Starting test process...")
        print("📊 Configuration:")
        print(f"  - Save dir: {self.save_dir}")
        print(f"  - Haystack dir: {self.haystack_dir}")
        print(f"  - Context lengths: {self.context_lengths}")
        print(f"  - Depth percentages: {self.document_depth_percents}")

        if not os.path.exists(self.save_dir):
            print(f"📁 Creating save directory: {self.save_dir}")
            os.makedirs(self.save_dir)

        try:
            asyncio.run(self.run_test())
            print("🎉 Test completed successfully")
        except Exception as e:
            print(f"❌ Test failed: {str(e)}")
            raise


if __name__ == "__main__":
    try:
        print("📝 Loading configuration...")
        config_path = Path(__file__).resolve().parent / CONF_FILE
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)

        print("🔧 Creating Prompter instance...")
        ht = Prompter(
            needle=config["prompt"]["needle"],
            haystack_dir=config["prompt"][
                "haystack_dir"
            ],  # Fix: Access haystack_dir from prompt config
            retrieval_question=config["prompt"]["retrieval_question"],
            context_lengths_min=config["context"]["min_len"],
            context_lengths_max=config["context"]["max_len"],
            context_lengths_num_intervals=config["context"]["interval"],
            context_lengths=config["context"]["manually_select_list"],
            document_depth_percent_min=config["document_depth"]["min_percent"],
            document_depth_percent_max=config["document_depth"]["max_percent"],
            document_depth_percent_intervals=config["document_depth"][
                "interval"
            ],
            document_depth_percents=config["document_depth"][
                "manually_select_list"
            ],
            document_depth_percent_interval_type=config["document_depth"][
                "interval_type"
            ],
            tokenizer_type=config["tokenizer"]["tokenizer_type"],
            model_name=config["tokenizer"]["model_name"],
            save_dir=config["save_dir"],
        )

        ht.start_test()

    except Exception as e:
        print(f"🚨 Error initializing Prompter: {str(e)}")
