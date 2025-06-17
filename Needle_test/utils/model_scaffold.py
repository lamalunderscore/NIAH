"""Scaffold classes to load models and interact with them in a generic way."""

from abc import ABC, abstractmethod
from glob import glob
from pathlib import Path
from typing import Any, Literal

import kagglehub
import sentencepiece as spm
import torch
from accelerate import (
    infer_auto_device_map,
    load_checkpoint_and_dispatch,
)
from accelerate.utils import get_balanced_memory
from transformers import AutoTokenizer

from jamba import JambaForCausalLM
from recorder import AttentionRecorder
from recurrentgemma.common import GriffinConfig, Preset
from recurrentgemma.torch.griffin import Griffin
from recurrentgemma.torch.sampler import Sampler


class Model(ABC):
    """Abstract base class for all models and providers.

    Attributes:
        model : The loaded interaction object.

    """

    def __init__(
        self,
        model_id: str,
        tokenizer_id: str | None = None,
        max_tokens: int = 40,
        dtype: torch.dtype | None = None,
        device: str | None = None,
    ):
        self.model_id = model_id
        self.tokenizer_id = tokenizer_id
        self.model: Any = None
        self.max_tokens = max_tokens
        self.device = device
        self.dtype = dtype
        self._model_init(model_id, tokenizer_id)
        self._attention_recorders: dict[str, AttentionRecorder] | None = None

    def __call__(self, inputs, **kwargs):
        return self.get_output(inputs, **kwargs)

    def get_recorded_attention(self) -> dict[str, list[torch.Tensor | Any] | torch.Tensor] | None:
        if isinstance(self._attention_recorders, dict):
            recorded_attention_dict: dict[str, list[torch.Tensor | None] | torch.Tensor] = {}
            for name, recorder in self._attention_recorders.items():
                assert isinstance(recorder, AttentionRecorder)
                recorded_attention = recorder.get()
                if recorded_attention is None or (isinstance(recorded_attention, list) and not recorded_attention):
                    print(
                        f"Warning: No data found when getting recorded attention from {name}. Either the model did not run yet, "
                        "attention recording was not initialized (use `self.model.enable_attention_recording()`), or the initialization failed. "
                        "Omitting this recorder."
                    )
                    continue
                recorded_attention_dict[name] = recorded_attention
            return recorded_attention_dict
        elif self._attention_recorders is None:
            print(
                "Warning: self.attention_recorders is None, consider enabling attention recording: `self.enable_attention_recording()`."
            )
            return None
        else:
            raise TypeError("self._attention_recorders has an invalid type.")

    def enable_attention_recording(
        self,
        record_mode: Literal["first", "last", "all"] = "first",
    ):
        """Enable attention recording."""
        attention_modules = self.model.attention_modules
        assert attention_modules is not None, "No attention blocks found. Error in model initialization."
        self._attention_recorders = {}
        for name in attention_modules:
            recorder = AttentionRecorder(name, record_mode=record_mode)
            attention_modules[name].attention_recorder = recorder
            self._attention_recorders[name] = recorder
        print(f"enabled attention recording. Initialized {len(self._attention_recorders.keys())} recorders.")

    def disable_attention_recording(self):
        """Disable attention recording."""
        attention_modules = self.model.attention_modules
        assert attention_modules is not None, "No attention blocks found. Error in model initialization."
        if self._attention_recorders is not None:
            for name in attention_modules:
                attention_modules[name].attention_recorder = None
            self._attention_recorders = None
            print("Disabled attention recording")
        else:
            print("Warning: Attention recording was already disabled. Ignore this message if it was expected.")

    def _set_sparse_attributes(
        self,
        k: int | None = None,
        metric: str | None = None,
        prefill: bool | None = False,
    ):
        attention_modules = self.model.attention_modules
        assert attention_modules is not None, "No attention blocks found. Error in model initialization."
        for layer in attention_modules.values():
            layer.topk_heads = k
            layer.sparsity_metric = metric
            layer.sparsity_prefill = prefill

    def enable_sparsification(self, k: int = 2, metric="entropy", prefill: bool = False):
        """Enable attention head sparsification.

        Specify k value, norm, and if it should be applied during prefill.
        """
        self._set_sparse_attributes(k, metric, prefill)

    def disable_sparsification(self):
        """Disable attention head sparsification."""
        self._set_sparse_attributes()
        print("disabled sparsification")

    @abstractmethod
    def _model_init(self, model_id: str, tokenizer_id: str | None): ...

    @abstractmethod
    def get_output(self, *args: Any, **kwargs: Any) -> list[str]: ...

    @abstractmethod
    def load_model(self, *args: Any, **kwargs: Any) -> Any: ...


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

    def _model_init(self, model_id, tokenizer_id=None):
        self.model = self.load_model(model_id)
        if tokenizer_id is None:
            tokenizer_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_id)

    def get_output(self, messages, **kwargs):
        input_ids = self.tokenizer.apply_chat_template(  # TODO, make it respect the base model boolean
            messages,
            add_generation_prompt=True,
            padding=True,
            return_tensors="pt",
        ).to(self.model.device)

        outputs = self.model.generate(input_ids, max_new_tokens=self.max_tokens)
        out_data: list[str] = [
            self.tokenizer.decode(gen[input_ids.shape[-1] :], skip_special_tokens=True) for gen in outputs
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
                **kwargs,  # type: ignore
            )

            max_memory = get_balanced_memory(
                model,
                no_split_module_classes=no_split_module_classes,
                dtype=kwargs.get("torch_dtype"),  # type: ignore
            )
            print(f"max_memory: {max_memory}")

            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
                dtype=kwargs.get("torch_dtype"),  # type: ignore
            )

            del model
            torch.cuda.empty_cache()

        # Load actual model with device map
        model = model_cls.from_pretrained(model_id, device_map=device_map, **kwargs)  # type: ignore
        model.eval()
        return model


class RecurrentGemmaKaggle(Model):
    """Class to scaffold the RG Kaggle implementation."""

    def _model_init(self, model_id: str, tokenizer_id: str | None = None):
        self.model = self.load_model(model_id)

    def get_output(self, inputs: str | list[str], **gen_kwargs):
        if isinstance(inputs, str):
            inputs = [inputs]
        outputs = self.sampler(input_strings=inputs, total_generation_steps=self.max_tokens, **gen_kwargs)
        out_data = outputs.text
        return out_data

    def load_model(
        self,
        model_id,
        device: str | None = None,
        dtype: torch.dtype | None = None,
    ):
        if device is None:
            device = self.device
        if dtype is None:
            dtype = self.dtype

        model_dir = Path(kagglehub.model_download(model_id))
        model_path = Path(glob(str(model_dir / "*.pt"))[0])
        tokenizer_path = str(model_dir / "tokenizer.model")
        print(f"type tokenizer_path: {type(tokenizer_path)}")

        params = torch.load(model_path)
        params = {k: v.to(device=device, dtype=dtype) for k, v in params.items()}
        preset = Preset.RECURRENT_GEMMA_2B_V1 if "2b" in model_path.name else Preset.RECURRENT_GEMMA_9B_V1
        model_config = GriffinConfig.from_torch_params(params, preset=preset)
        model = Griffin(model_config, device=device, dtype=dtype)
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
            model = load_checkpoint_and_dispatch(model, checkpoint=model_path, device_map=device_map)
        else:
            print("Using single-GPU setup")
            model.load_state_dict(params)
        assert isinstance(model, Griffin), "This implementation expects Griffin Module."
        model.eval()

        vocab = spm.SentencePieceProcessor()
        vocab.Load(tokenizer_path)

        self.sampler = Sampler(model=model, vocab=vocab)
        return model
