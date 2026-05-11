import torch
import torch.nn as nn
from tqdm import tqdm


# ============================================================
# BASE PATCH SYSTEM
# ============================================================

class BasePatch:
    @classmethod
    def get_linear_tags(cls):
        raise NotImplementedError

    @classmethod
    def patch_nonlinearlayers(cls, model, patch_fct, verbose=True):
        raise NotImplementedError

    @classmethod
    def patch_linearlayers(cls, model, patch_fct, patch_params, verbose=True):
        raise NotImplementedError


# ============================================================
# ARCHITECTURE DETECTION
# ============================================================

class ArchitectureDetector:

    @staticmethod
    def get_base_model(model):

        if hasattr(model, "model"):
            return model.model

        if hasattr(model, "transformer"):
            return model.transformer

        if hasattr(model, "gpt_neox"):
            return model.gpt_neox

        raise ValueError(
            f"Unsupported architecture: {type(model)}"
        )

    @staticmethod
    def get_layers(base_model):

        if hasattr(base_model, "layers"):
            return base_model.layers

        if hasattr(base_model, "h"):
            return base_model.h

        raise ValueError(
            "Cannot locate transformer layers"
        )


# ============================================================
# QUANTIZED WRAPPER
# ============================================================

class QuantLinear(nn.Module):

    def __init__(self, linear, bits=4):
        super().__init__()

        self.bits = bits
        self.in_features = linear.in_features
        self.out_features = linear.out_features

        # Fake quantization placeholder
        # Replace with HQQ/GPTQ/AWQ later
        self.weight = nn.Parameter(
            linear.weight.data.clone()
        )

        self.bias = None

        if linear.bias is not None:
            self.bias = nn.Parameter(
                linear.bias.data.clone()
            )

    def forward(self, x):

        # Simulated quantization
        scale = torch.max(torch.abs(self.weight)) / (
            2 ** (self.bits - 1)
        )

        qweight = torch.round(
            self.weight / scale
        ) * scale

        return torch.nn.functional.linear(
            x,
            qweight,
            self.bias
        )


# ============================================================
# PATCH FUNCTIONS
# ============================================================

def quant_patch(module, params=None):

    if params is None:
        params = {}

    bits = params.get("bits", 4)

    if isinstance(module, nn.Linear):

        return QuantLinear(
            module,
            bits=bits
        )

    return module


def nonlinear_patch(module):

    # Placeholder for:
    # - fused norm
    # - activation replacement
    # - flash rotary
    # - memory optimization

    return module


# ============================================================
# LLAMA PATCHER
# ============================================================

class LLamaPatch(BasePatch):

    @classmethod
    def get_linear_tags(cls):

        return [
            "self_attn.q_proj",
            "self_attn.k_proj",
            "self_attn.v_proj",
            "self_attn.o_proj",
            "mlp.gate_proj",
            "mlp.up_proj",
            "mlp.down_proj",
        ]

    @classmethod
    @torch.no_grad()
    def patch_nonlinearlayers(
        cls,
        model,
        patch_fct,
        verbose=True
    ):

        base_model = ArchitectureDetector.get_base_model(model)

        if hasattr(model, "lm_head"):
            model.lm_head = patch_fct(model.lm_head)

        if hasattr(base_model, "embed_tokens"):
            base_model.embed_tokens = patch_fct(
                base_model.embed_tokens
            )

        if hasattr(base_model, "norm"):
            base_model.norm = patch_fct(
                base_model.norm
            )

        layers = ArchitectureDetector.get_layers(
            base_model
        )

        for i in tqdm(
            range(len(layers)),
            disable=not verbose,
            desc="Patching nonlinear layers"
        ):

            layer = layers[i]

            try:

                if hasattr(layer.self_attn, "rotary_emb"):
                    layer.self_attn.rotary_emb = patch_fct(
                        layer.self_attn.rotary_emb
                    )

                if hasattr(layer.mlp, "act_fn"):
                    layer.mlp.act_fn = patch_fct(
                        layer.mlp.act_fn
                    )

                if hasattr(layer, "input_layernorm"):
                    layer.input_layernorm = patch_fct(
                        layer.input_layernorm
                    )

                if hasattr(layer, "post_attention_layernorm"):
                    layer.post_attention_layernorm = patch_fct(
                        layer.post_attention_layernorm
                    )

            except Exception as e:

                print(
                    f"[Nonlinear Patch Error] "
                    f"Layer {i}: {e}"
                )

    @classmethod
    @torch.no_grad()
    def patch_linearlayers(
        cls,
        model,
        patch_fct,
        patch_params,
        verbose=True
    ):

        base_model = ArchitectureDetector.get_base_model(model)

        layers = ArchitectureDetector.get_layers(
            base_model
        )

        for i in tqdm(
            range(len(layers)),
            disable=not verbose,
            desc="Patching linear layers"
        ):

            layer = layers[i]

            try:

                # ==========================
                # ATTENTION
                # ==========================

                layer.self_attn.q_proj = patch_fct(
                    layer.self_attn.q_proj,
                    patch_params.get(
                        "self_attn.q_proj",
                        {}
                    )
                )

                layer.self_attn.k_proj = patch_fct(
                    layer.self_attn.k_proj,
                    patch_params.get(
                        "self_attn.k_proj",
                        {}
                    )
                )

                layer.self_attn.v_proj = patch_fct(
                    layer.self_attn.v_proj,
                    patch_params.get(
                        "self_attn.v_proj",
                        {}
                    )
                )

                layer.self_attn.o_proj = patch_fct(
                    layer.self_attn.o_proj,
                    patch_params.get(
                        "self_attn.o_proj",
                        {}
                    )
                )

                # ==========================
                # MLP
                # ==========================

                layer.mlp.gate_proj = patch_fct(
                    layer.mlp.gate_proj,
                    patch_params.get(
                        "mlp.gate_proj",
                        {}
                    )
                )

                layer.mlp.up_proj = patch_fct(
                    layer.mlp.up_proj,
                    patch_params.get(
                        "mlp.up_proj",
                        {}
                    )
                )

                layer.mlp.down_proj = patch_fct(
                    layer.mlp.down_proj,
                    patch_params.get(
                        "mlp.down_proj",
                        {}
                    )
                )

            except Exception as e:

                print(
                    f"[Linear Patch Error] "
                    f"Layer {i}: {e}"
                )


# ============================================================
# AUTO PATCHER
# ============================================================

class AutoPatch:

    PATCHERS = {
        "llama": LLamaPatch,
    }

    @classmethod
    def detect_architecture(cls, model):

        model_name = str(type(model)).lower()

        if "llama" in model_name:
            return "llama"

        return "llama"

    @classmethod
    def patch_model(
        cls,
        model,
        quant_bits=4,
        verbose=True
    ):

        arch = cls.detect_architecture(model)

        patcher = cls.PATCHERS[arch]

        print(f"[LLMSTER] Detected architecture: {arch}")

        # ====================================================
        # QUANT CONFIG
        # ====================================================

        patch_params = {

            "self_attn.q_proj": {"bits": quant_bits},
            "self_attn.k_proj": {"bits": quant_bits},
            "self_attn.v_proj": {"bits": quant_bits},
            "self_attn.o_proj": {"bits": quant_bits},

            "mlp.gate_proj": {"bits": quant_bits},
            "mlp.up_proj": {"bits": quant_bits},
            "mlp.down_proj": {"bits": quant_bits},
        }

        # ====================================================
        # APPLY PATCHES
        # ====================================================

        patcher.patch_nonlinearlayers(
            model,
            nonlinear_patch,
            verbose=verbose
        )

        patcher.patch_linearlayers(
            model,
            quant_patch,
            patch_params,
            verbose=verbose
        )

        print(
            f"[LLMSTER] Model patched successfully "
            f"with {quant_bits}-bit quantization"
        )

        return model


# ============================================================
# VRAM DETECTOR
# ============================================================

class DeviceManager:

    @staticmethod
    def get_best_device():

        if torch.cuda.is_available():

            gpu_name = torch.cuda.get_device_name(0)

            total_vram = (
                torch.cuda.get_device_properties(0)
                .total_memory
                / 1024**3
            )

            print(
                f"[GPU] {gpu_name} "
                f"({total_vram:.2f} GB)"
            )

            return "cuda"

        print("[GPU] CUDA unavailable")

        return "cpu"

    @staticmethod
    def recommend_quantization():

        if not torch.cuda.is_available():
            return 2

        total_vram = (
            torch.cuda.get_device_properties(0)
            .total_memory
            / 1024**3
        )

        if total_vram > 40:
            return 8

        if total_vram > 20:
            return 4

        return 2


# ============================================================
# EXAMPLE USAGE
# ============================================================

if __name__ == "__main__":

    from transformers import AutoModelForCausalLM

    MODEL_NAME = "meta-llama/Llama-2-7b-hf"

    device = DeviceManager.get_best_device()

    bits = DeviceManager.recommend_quantization()

    print(f"[LLMSTER] Recommended quantization: {bits}-bit")

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.float16,
        device_map="auto"
    )

    model = AutoPatch.patch_model(
        model,
        quant_bits=bits,
        verbose=True
    )

    print("[LLMSTER] Ready for inference.")
