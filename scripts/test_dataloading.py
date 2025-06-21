from open_instruct.finetune import main
import importlib

from typing import Any
from unittest.mock import patch, Mock
from accelerate import Accelerator

def patch_target_module(
    to_patch: str,
    replace_with: Any,
    target_module: str = None,
):
    to_patch = to_patch.split(".")
    assert len(to_patch) > 1, "must have an object to patch"

    to_patch, obj_name_to_patch = to_patch[:-1], to_patch[-1]
    to_patch = ".".join(to_patch)
    source = importlib.import_module(to_patch)
    original_obj = getattr(source, obj_name_to_patch)
    setattr(source, obj_name_to_patch, replace_with)

    if target_module is not None:
        # reload and this should get the patched object
        target_module = importlib.import_module(target_module)
        importlib.reload(target_module)

        # replace it
        setattr(source, obj_name_to_patch, original_obj)

class DummyAccelerator:

    def __init__(*args, **kwargs):
        pass


from transformers import AutoModelForCausalLM, AutoConfig
from accelerate import init_empty_weights
from accelerate.state import AcceleratorState
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

# MODEL_CONFIG = None
# def from_pretrained(model_name_or_path, *args, **kwargs):
#     global MODEL_CONFIG
#     MODEL_CONFIG = AutoConfig.from_pretrained(model_name_or_path)
#     return Mock()
from types import MethodType

STORE = []

def built_from_pretrained():

    def forward(self, input_ids, *args, **kwargs):
        STORE.append(input_ids)
        return CausalLMOutputWithPast(
            loss=torch.tensor(0.),
        )

    old_func = AutoModelForCausalLM.from_pretrained
    def _from_pretrained(model_name_or_path, *args, **kwargs):
        with init_empty_weights():
            model = old_func(
                model_name_or_path, *args, **kwargs,
            )
            model.forward = MethodType(forward, model)
            return model

    return _from_pretrained

patch_transformers = patch.multiple(
    AutoModelForCausalLM,
    # from_pretrained=Mock(),
    from_pretrained=built_from_pretrained(),
)

def accelerate_prepare(self, model, optimizer, dataloader, scheduler):
    self.state = Mock() # sneak it in

    def set_epoch(self, *args):
        pass

    dataloader.set_epoch = MethodType(set_epoch, dataloader)
    return model, optimizer, dataloader, scheduler

import torch

patch_accelerate = patch.multiple(
    Accelerator,
    wait_for_everyone=Mock(),
    prepare=accelerate_prepare,
    num_processes=1,
    device=torch.device('cuda'),
    backward=Mock(),
    sync_gradients=False,
)

# patch_accelerate_state = patch.multiple(
#     AcceleratorState,
#     fsdp_plugin=Mock(),
# )

# import deepspeed
# import torch
# 
# def deepspeed_enter(*args, **kargs):
#     MODEL_CONFIG.
#     return torch.nn.Embedding(
# 
#     )
# 
# patch_deepspeed = patch.multiple(
#     deepspeed.zero.GatheredParameters,
#     __enter__=
# )
