"""Scaffold classes to load models and interact with them in a generic way."""

from abc import ABC, abstractmethod
from collections import OrderedDict
from glob import glob
from pathlib import Path
from typing import Any, Literal, overload

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

    @abstractmethod
    def _model_init(self, model_id: str, tokenizer_id: str | None): ...

    @abstractmethod
    def find_needle(self, needle: str, prompt: str | dict[str, str]) -> list[int] | None: ...

    @abstractmethod
    def get_output(self, *args: Any, **kwargs: Any) -> list[str]: ...

    @abstractmethod
    def load_model(self, *args: Any, **kwargs: Any) -> Any: ...

    def __call__(self, inputs, **kwargs):
        return self.get_output(inputs, **kwargs)

    def _get_single_mode_attention(self, get_mode: Literal["gen", "prefill"]):
        if isinstance(self._attention_recorders, dict):
            recorded_attention_dict: dict[str, list[torch.Tensor | None] | torch.Tensor] = {}
            for name, recorder in self._attention_recorders.items():
                assert isinstance(recorder, AttentionRecorder)
                recorded_attention = recorder.get_clear(get_mode)
                if recorded_attention is None or (
                    recorder.record_mode == "all" and not recorded_attention
                ):
                    print(
                        f"Warning: No data found when getting recorded attention from {name}. Either the model did not run yet, "
                        "attention recording was not initialized (use `self.model.enable_attention_recording()`), or the initialization failed. "
                        "Omitting this recorder. (in Model.get_recorded_attention)"
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

    @overload
    def get_recorded_attention(
        self, get_mode: Literal["prefill"] = "prefill"
    ) -> dict[str, list[torch.Tensor | Any] | torch.Tensor] | None: ...

    @overload
    def get_recorded_attention(
        self, get_mode: Literal["gen"] = "gen"
    ) -> dict[str, list[torch.Tensor | Any] | torch.Tensor] | None: ...

    @overload
    def get_recorded_attention(
        self, get_mode: Literal["both"] = "both"
    ) -> tuple[
        dict[str, list[torch.Tensor | Any] | torch.Tensor] | None,
        dict[str, list[torch.Tensor | Any] | torch.Tensor] | None,
    ]: ...

    def get_recorded_attention(
        self, get_mode: Literal["gen", "prefill", "both"] = "both"
    ) -> (
        tuple[
            dict[str, list[torch.Tensor | Any] | torch.Tensor] | None,
            dict[str, list[torch.Tensor | Any] | torch.Tensor] | None,
        ]
        | dict[str, list[torch.Tensor | Any] | torch.Tensor]
        | None
    ):
        if get_mode == "both":
            return (
                self._get_single_mode_attention("gen"),
                self._get_single_mode_attention("prefill"),
            )
        else:
            return self._get_single_mode_attention(get_mode)

    def enable_attention_recording(
        self,
        record_mode: Literal["first", "last", "all"] = "first",
    ):
        """Enable attention recording."""
        attention_modules = self.model.attention_modules
        assert attention_modules is not None, (
            "No attention blocks found. Error in model initialization."
        )
        self._attention_recorders = {}
        for name in attention_modules:
            recorder = AttentionRecorder(name, record_mode=record_mode)
            attention_modules[name].attention_recorder = recorder
            self._attention_recorders[name] = recorder
        print(
            f"Enabled attention recording. Initialized {len(self._attention_recorders.keys())} recorders."
        )

    def disable_attention_recording(self):
        """Disable attention recording."""
        attention_modules = self.model.attention_modules
        assert attention_modules is not None, (
            "No attention blocks found. Error in model initialization."
        )
        if self._attention_recorders is not None:
            for name in attention_modules:
                attention_modules[name].attention_recorder = None
            self._attention_recorders = None
            print("Disabled attention recording")
        else:
            print(
                "Warning: Attention recording was already disabled. Ignore this message if it was expected."
            )

    def _set_sparse_head_attributes(
        self,
        k: int | None = None,
        metric: str | None = None,
        prefill: bool | None = False,
    ):
        attention_modules = self.model.attention_modules
        assert attention_modules is not None, (
            "No attention blocks found. Error in model initialization."
        )
        for layer in attention_modules.values():
            layer.topk_heads = k
            layer.sparsity_metric = metric
            layer.sparsity_prefill = prefill

    def enable_head_sparsification(self, k: int = 2, metric="entropy", prefill: bool = False):
        """Enable attention head sparsification.

        Specify k value, norm, and if it should be applied during prefill.
        """
        self._set_sparse_head_attributes(k, metric, prefill)
        print(f"Enabled head sparsification with k={k}, metric={metric}, prefill={prefill}.")

    def disable_head_sparsification(self):
        """Disable attention head sparsification."""
        self._set_sparse_head_attributes()
        print("Disabled head sparsification.")

    def _set_manipulation_attributes(
        self,
        indices: list[int] | None = None,
        prefill_indices: list[int] | None = None,
        gen_mode: Literal["ommit", "only", "balanced"] | None = None,
        prefill_mode: Literal["ommit", "only", "balanced", "follow", "keep", "null"] | None = None,
    ):
        attention_modules = self.model.attention_modules
        assert attention_modules is not None, (
            "No attention blocks found. Error in model initialization."
        )
        for layer in attention_modules.values():
            layer.manipulate_gen_indices = indices
            layer.manipulate_prefill_indices = (
                indices if prefill_indices is None else prefill_indices
            )

            layer.manipulate_gen = gen_mode
            if prefill_mode == "follow":
                prefill_mode = gen_mode
            layer.manipulate_prefill = prefill_mode

    @overload
    def enable_weight_manipulation(
        self,
        indices: list[int],
        prefill_indices: list[int],
        *,
        gen_mode: Literal["ommit", "only", "balanced"] = "ommit",
        prefill_mode: Literal["ommit", "only", "balanced", "follow", "keep", "null"] = "keep",
    ): ...

    @overload
    def enable_weight_manipulation(
        self,
        indices: list[int],
        *,
        gen_mode: Literal["ommit", "only", "balanced"] = "ommit",
        prefill_mode: Literal["ommit", "only", "balanced", "follow", "keep", "null"] = "keep",
    ): ...

    def enable_weight_manipulation(
        self,
        indices: list[int],
        prefill_indices: list[int] | None = None,
        *,
        gen_mode: Literal["ommit", "only", "balanced"] = "ommit",
        prefill_mode: Literal["ommit", "only", "balanced", "follow", "keep", "null"] = "keep",
    ):
        """Enable attention head sparsification.

        Specify indices to be manipulated, mode, and if it should be applied during prefill.
        """
        self._set_manipulation_attributes(indices, prefill_indices, gen_mode, prefill_mode)
        print(
            f"Enabled weight manipulation on tokens {indices[0]}-{indices[-1]} token, gen_mode={gen_mode}, prefill_mode={prefill_mode}."
        )

    def disable_weight_manipulation(self):
        """Disable attention head sparsification."""
        self._set_manipulation_attributes()
        print("Disabled weight manipulation.")

    def _find_needle(self, needle: list[int], haystack: list[int]):
        needle_len = len(needle)
        input_len = len(haystack)
        for i in range(input_len - needle_len + 1):
            haystack_batch = haystack[i : i + needle_len]
            if haystack_batch == needle:
                return list(range(i, i + needle_len))
        return None


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

    def find_needle(self, needle: str, prompt: str | dict[str, str]):
        raise NotImplementedError("Not yet implemented.")

    def get_output(self, messages, **kwargs):
        input_ids = (
            self.tokenizer.apply_chat_template(  # TODO, make it respect the base model boolean
                messages,
                add_generation_prompt=True,
                padding=True,
                return_tensors="pt",
            ).to(self.model.device)
        )

        outputs = self.model.generate(input_ids, max_new_tokens=self.max_tokens)
        out_data: list[str] = [
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
                **kwargs,  # type: ignore
            )

            max_memory = get_balanced_memory(
                model,
                no_split_module_classes=no_split_module_classes,
                dtype=kwargs.get("torch_dtype"),  # type: ignore
            )
            print(f"max_memory after correction: {max_memory}")

            device_map = infer_auto_device_map(
                model,
                max_memory=max_memory,
                no_split_module_classes=no_split_module_classes,
                dtype=kwargs.get("torch_dtype"),  # type: ignore
            )

            device_map = layer_mapping = OrderedDict(
                [
                    ("model.embed_tokens", 0),
                    ("model.layers.0", 0),
                    ("model.layers.1", 0),
                    ("model.layers.2", 0),
                    ("model.layers.3", 1),
                    ("model.layers.4", 1),
                    ("model.layers.5", 1),
                    ("model.layers.6", 1),
                    ("model.layers.7", 1),
                    ("model.layers.8", 2),
                    ("model.layers.9", 2),
                    ("model.layers.10", 2),
                    ("model.layers.11", 2),
                    ("model.layers.12", 2),
                    ("model.layers.13", 3),
                    ("model.layers.14", 3),
                    ("model.layers.15", 3),
                    ("model.layers.16", 3),
                    ("model.layers.17", 3),
                    ("model.layers.18", 4),
                    ("model.layers.19", 4),
                    ("model.layers.20", 4),
                    ("model.layers.21", 4),
                    ("model.layers.22", 5),
                    ("model.layers.23", 5),
                    ("model.layers.24", 5),
                    ("model.layers.25", 5),
                    ("model.layers.26", 6),
                    ("model.layers.27", 6),
                    ("model.layers.28", 6),
                    ("model.layers.29", 6),
                    ("model.layers.30", 7),
                    ("model.layers.31", 7),
                    ("model.final_layernorm", 7),
                    ("lm_head", 7),
                ]
            )
            print(f"device map: {device_map}")

            del model
            # raise NotImplementedError("Stop script.")

        # Load actual model with device map
        model = model_cls.from_pretrained(model_id, device_map=device_map, **kwargs)  # type: ignore
        model.eval()
        return model


class RecurrentGemmaKaggle(Model):
    """Class to scaffold the RG Kaggle implementation."""

    def _model_init(self, model_id: str, tokenizer_id: str | None = None):
        self.model, self.tokenizer = self.load_model(model_id)

    def find_needle(self, needle: str, prompt: str | dict[str, str]):
        needle_ids = self.tokenizer.Encode(needle)[1:-1]
        prompt_ids = self.tokenizer.Encode(prompt)
        sequence = self._find_needle(needle_ids, prompt_ids)

        return sequence

    def get_output(self, inputs: str | list[str], **gen_kwargs):
        if isinstance(inputs, str):
            inputs = [inputs]
        outputs = self.sampler(
            input_strings=inputs,
            total_generation_steps=self.max_tokens,
            end_sampling_at_eos_token=True,
            **gen_kwargs,
        )
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
        preset = (
            Preset.RECURRENT_GEMMA_2B_V1
            if "2b" in model_path.name
            else Preset.RECURRENT_GEMMA_9B_V1
        )
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
            model = load_checkpoint_and_dispatch(
                model, checkpoint=str(model_path), device_map=device_map
            )
        else:
            print("Using single-GPU setup")
            model.load_state_dict(params)
        assert isinstance(model, Griffin), "This implementation expects Griffin Module."
        model.eval()

        vocab = spm.SentencePieceProcessor()
        vocab.Load(str(tokenizer_path))

        self.sampler = Sampler(model=model, vocab=vocab)
        return model, vocab
