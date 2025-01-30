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

load_dotenv()

class Prompter:
    """
    This class is used to test the LLM Needle Haystack with improved tokenizer handling.
    """
    def __init__(self,
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
                 print_ongoing_status=True):
        """Initialization of the prompter class with enhanced error handling."""
        # Validate essential inputs
        if not all([needle, haystack_dir, retrieval_question]):
            raise ValueError("Needle, haystack_dir, and retrieval_question must all be provided.")
        
        # Store basic parameters
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

        # Validate model name
        if not isinstance(self.model_name, str):
            raise ValueError(f"model_name must be a string, got {type(self.model_name)}: {self.model_name}")
        print(f"🔍 Debug: model_name = {self.model_name}")

        # Handle context lengths
        self._setup_context_lengths(
            context_lengths,
            context_lengths_min,
            context_lengths_max,
            context_lengths_num_intervals
        )

        # Handle document depth percentages
        self._setup_document_depth_percents(
            document_depth_percents,
            document_depth_percent_min,
            document_depth_percent_max,
            document_depth_percent_intervals,
            document_depth_percent_interval_type
        )

        # Initialize tokenizer
        self._initialize_tokenizer()

    def _setup_context_lengths(self, context_lengths, min_len, max_len, num_intervals):
        """Set up context lengths with validation."""
        if context_lengths is None:
            if any(x is None for x in [min_len, max_len, num_intervals]):
                raise ValueError(
                    "Either provide context_lengths list or all of: "
                    "context_lengths_min, context_lengths_max, context_lengths_num_intervals"
                )
            self.context_lengths = np.round(
                np.linspace(min_len, max_len, num=num_intervals, endpoint=True)
            ).astype(int)
        else:
            self.context_lengths = np.array(context_lengths)

    def _setup_document_depth_percents(self, depth_percents, min_percent, 
                                     max_percent, intervals, interval_type):
        """Set up document depth percentages with validation."""
        if depth_percents is None:
            if any(x is None for x in [min_percent, max_percent, intervals]):
                raise ValueError(
                    "Either provide document_depth_percents list or all of: "
                    "document_depth_percent_min, document_depth_percent_max, document_depth_percent_intervals"
                )
            
            if interval_type == "linear":
                self.document_depth_percents = np.round(
                    np.linspace(min_percent, max_percent, num=intervals, endpoint=True)
                ).astype(int)
            elif interval_type == "sigmoid":
                self.document_depth_percents = [
                    self.logistic(x) for x in 
                    np.linspace(min_percent, max_percent, intervals)
                ]
            else:
                raise ValueError(
                    "document_depth_percent_interval_type must be either 'linear' or 'sigmoid'"
                )
        else:
            self.document_depth_percents = np.array(depth_percents)

    def _initialize_tokenizer(self):
        """Initialize the tokenizer with enhanced error handling."""
        try:
            if self.tokenizer_type == "OpenAI":
                if self.model_name is None:
                    raise ValueError("OpenAI tokenizer requires a model name")
                self.enc = tiktoken.encoding_for_model(self.model_name)
                
            elif self.tokenizer_type == "Anthropic":
                self.enc = Anthropic().get_tokenizer()
                
            elif self.tokenizer_type == "Huggingface":
                self._initialize_huggingface_tokenizer()
                
            else:
                raise ValueError(
                    f"Unsupported tokenizer_type: {self.tokenizer_type}. "
                    "Must be 'OpenAI', 'Anthropic', or 'Huggingface'"
                )
                
            print(f"✅ Successfully initialized {self.tokenizer_type} tokenizer")
            
        except Exception as e:
            raise ValueError(f"Failed to initialize {self.tokenizer_type} tokenizer: {str(e)}")

    def _initialize_huggingface_tokenizer(self):
        """Initialize HuggingFace tokenizer with proper error handling."""
        # Check if it's a local path
        is_local = os.path.exists(self.model_name)
        
        if is_local:
            model_path = Path(self.model_name)
            required_files = ['tokenizer.json', 'tokenizer_config.json']
            
            missing_files = [f for f in required_files if not (model_path / f).exists()]
            if missing_files:
                raise ValueError(
                    f"Missing required tokenizer files in {self.model_name}: {', '.join(missing_files)}"
                )
            
            print(f"📂 Loading local tokenizer from {self.model_name}")
        else:
            print(f"🌐 Loading tokenizer from HuggingFace Hub: {self.model_name}")

        try:
            # For Gemma models, we need to handle the tokenizer initialization differently
            if 'gemma' in self.model_name.lower():
                from transformers import AutoTokenizer
                self.enc = AutoTokenizer.from_pretrained(
                    self.model_name,
                    use_fast=True,  # Changed to True for Gemma
                    local_files_only=is_local,
                    trust_remote_code=True
                )
            else:
                self.enc = AutoTokenizer.from_pretrained(
                    self.model_name,
                    trust_remote_code=True,
                    use_fast=False,
                    local_files_only=is_local
                )
        except Exception as e:
            # Add more specific error handling for common issues
            if "not a string" in str(e):
                raise ValueError(
                    f"Tokenizer initialization failed for {self.model_name}. "
                    "This might be due to incompatible tokenizer files. "
                    "Please ensure you're using the correct model type and all necessary files are present."
                )
            else:
                raise ValueError(
                    f"Failed to load HuggingFace tokenizer: {str(e)}\n"
                    f"For local models, ensure the path contains valid tokenizer files.\n"
                    f"For remote models, ensure the model ID is correct and you have internet access."
                )

    @staticmethod
    def logistic(x):
        """Compute logistic function for sigmoid distribution."""
        return int(100 / (1 + np.exp(-0.1 * (x - 50))))

    def start_test(self):
        """Start the testing process with debug logging."""
        print(f"🔍 Starting test with following configuration:")
        print(f"  - Save directory: {self.save_dir}")
        print(f"  - Haystack directory: {self.haystack_dir}")
        print(f"  - Context lengths: {self.context_lengths}")
        print(f"  - Document depth percentages: {self.document_depth_percents}")
        
        # Create save directory if it doesn't exist
        if not os.path.exists(self.save_dir):
            print(f"📁 Creating save directory: {self.save_dir}")
            os.makedirs(self.save_dir)
        
        try:
            # Load haystack content
            haystack_files = glob.glob(os.path.join(self.haystack_dir, "*"))
            print(f"📚 Found {len(haystack_files)} files in haystack directory")
            
            for context_length in self.context_lengths:
                for depth_percent in self.document_depth_percents:
                    prompt_data = {
                        "model": self.model_name,
                        "tokenizer": self.tokenizer_type,
                        "context_length": int(context_length),
                        "depth_percent": int(depth_percent),
                        "needle": self.needle,
                        "retrieval_question": self.retrieval_question,
                        "content": [
                            {"role": "system", "content": "You are a helpful assistant."},
                            {"role": "user", "content": self.retrieval_question}
                        ]
                    }
                    
                    # Generate filename
                    filename = f"{self.tokenizer_type}_{context_length}_{depth_percent}_prompts.json"
                    save_path = os.path.join(self.save_dir, filename)
                    
                    print(f"💾 Saving prompt to: {save_path}")
                    print(f"  - Context length: {context_length}")
                    print(f"  - Depth percent: {depth_percent}")
                    
                    # Save the prompt
                    try:
                        with open(save_path, 'w') as f:
                            json.dump(prompt_data, f, indent=2)
                        print(f"✅ Successfully saved prompt to {save_path}")
                    except Exception as e:
                        print(f"❌ Failed to save prompt to {save_path}: {str(e)}")
            
            print("🎉 Saving completed!")
            
        except Exception as e:
            print(f"❌ Error during test execution: {str(e)}")
            raise


if __name__ == '__main__':
    with open('config-prompt.yaml', 'r') as file:
        config = yaml.safe_load(file)  # Using safe_load instead of load for security

    try:
        ht = Prompter(
            needle=config['prompt']['needle'],
            haystack_dir=config['prompt']['haystack_dir'],
            retrieval_question=config['prompt']['retrieval_question'],

            context_lengths_min=config['context']['min_len'],
            context_lengths_max=config['context']['max_len'],
            context_lengths_num_intervals=config['context']['interval'],
            context_lengths=config['context']['manually_select_list'],

            document_depth_percent_min=config['document_depth']['min_percent'],
            document_depth_percent_max=config['document_depth']['max_percent'],
            document_depth_percent_intervals=config['document_depth']['interval'],
            document_depth_percents=config['document_depth']['manually_select_list'],
            document_depth_percent_interval_type=config['document_depth']['interval_type'],

            tokenizer_type=config['tokenizer']['tokenizer_type'],
            model_name=config['tokenizer']['model_name'],

            save_dir=config['save_dir'],
        )

        ht.start_test()
        
    except Exception as e:
        print(f"🚨 Error initializing Prompter: {str(e)}")
