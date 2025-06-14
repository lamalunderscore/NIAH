"""Util classes to load models from different providers."""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Union

import sentencepiece as spm
import torch
from accelerate import (
    infer_auto_device_map,
    load_checkpoint_and_dispatch,
)
from accelerate.utils import get_balanced_memory
from transformers import AutoTokenizer

from jamba import JambaForCausalLM
from recurrentgemma import torch as recurrentgemma


class Model(ABC):
    """Abstract base class for all models and providers.

    Attributes:
        model : The loaded interaction object.

    """

    def __init__(
        self,
        model_path: str,
        tokenizer_path: str,
        device: str,
        max_tokens: Optional[int] = 40,
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        self.model = self.load_model(model_path, tokenizer_path, device)
        self.max_tokens = max_tokens

    def __call__(self, inputs, **kwargs):
        return self.get_output(inputs, **kwargs)

    @abstractmethod
    def get_output(self, inputs): ...

    @abstractmethod
    def load_model(self, model_path, tokenizer_path, device): ...


ALL_HUGGINGFACE_IMPLEMENTED = {
    "ai21labs/AI21-Jamba-Mini-1.6": (
        JambaForCausalLM,
        [
            "JambaMambaDecoderLayer",
            "JambaAttentionDecoderLayer",
            "JambaSparseMoeBlock",
        ],
        {
            "torch_dtype": torch.bfloat16,
            "attn_implementation": "eager",
            "use_mamba_kernels": False,
        },
    )
}


class Huggingface(Model):
    """Class to scaffold all Huggingface Models."""

    def __init__(
        self,
        model_id: str,
        tokenizer_id: str,
        device: str,
        max_tokens: Optional[int] = 40,
        dtype: Optional[torch.dtype] = torch.float32,
        # add is_base here
    ):
        self.model = self.load_model(model_id)
        self.max_tokens = max_tokens
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    def get_output(self, messages, **kwargs):
        input_ids = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",  # TODO, make it respext the base model boolean
        ).to(self.model.device)

        outputs = self.model.generate(input_ids, max_new_tokens=self.max_tokens)
        out_data = [
            self.tokenizer.decode(gen[input_ids.shape[-1] :], skip_special_tokens=True)
            for gen in outputs
        ]
        return out_data

    def load_model(self, model_id):
        try:
            model_cls, no_split_module_classes, kwargs = ALL_HUGGINGFACE_IMPLEMENTED[model_id]
        except KeyError:
            raise NotImplementedError(f"The model {model_id} is not implemented.")

        device_map = "auto"
        if torch.cuda.device_count() > 1:
            print("Using accelerate for multi-GPU inference")
            model = model_cls.from_pretrained(
                model_id,
                device_map="meta",  # Load structure only
                **kwargs,
            )

            max_memory = get_balanced_memory(
                model,
                no_split_module_classes=no_split_module_classes,
                dtype=kwargs.get("torch_dtype"),
            )
            print(f"max_memory: {max_memory}")

            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
                dtype=kwargs.get("torch_dtype"),
            )

            del model
            torch.cuda.empty_cache()

        # Load actual model with device map
        model = model_cls.from_pretrained(model_id, device_map=device_map, **kwargs)
        model.eval()
        return model


class RecurrentGemmaKaggle(Model):
    """Class to scaffold the RG Kaggle implementation."""

    def __init__(
        self,
        model_path: Union[str, Path],
        tokenizer_path: Union[str, Path],
        device: str,
        max_tokens: Optional[int] = 40,
        dtype: Optional[torch.dtype] = torch.float32,
    ):
        self.model = self.load_model(model_path, tokenizer_path, device)
        self.max_tokens = max_tokens

    def get_output(self, inputs, **gen_kwargs):
        outputs = self.sampler(
            input_strings=inputs, total_generation_steps=self.max_tokens, **gen_kwargs
        )
        out_data = outputs.text
        return out_data

    def load_model(self, model_path, tokenizer_path, device):
        dtype = torch.bfloat16

        if not isinstance(model_path, Path):
            model_path = Path(model_path)

        params = torch.load(model_path)
        params = {k: v.to(device=device, dtype=dtype) for k, v in params.items()}
        preset = (
            recurrentgemma.Preset.RECURRENT_GEMMA_2B_V1
            if "2b" in model_path.name
            else recurrentgemma.Preset.RECURRENT_GEMMA_9B_V1
        )
        model_config = recurrentgemma.GriffinConfig.from_torch_params(params, preset=preset)
        model = recurrentgemma.Griffin(model_config, device=device, dtype=dtype)
        if device == "cuda" and torch.cuda.device_count() > 1:
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
        model.eval()

        vocab = spm.SentencePieceProcessor()
        vocab.Load(tokenizer_path)

        self.sampler = recurrentgemma.Sampler(model=model, vocab=vocab)
        return model
