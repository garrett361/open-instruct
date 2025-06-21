import torch
import os

from unittest.mock import patch, Mock
from accelerate import Accelerator


from transformers import AutoModelForCausalLM
from accelerate import init_empty_weights
from transformers.modeling_outputs import CausalLMOutputWithPast

from types import MethodType

STORE = []

# builds the from_pretrained to pass through AutoModelForCausalLM
# - loads the model on the meta device

def built_from_pretrained():

    def forward(self, input_ids, *args, **kwargs):
        # - hook to capture the input_ids
        STORE.append({
            'input_ids': input_ids,
            'labels': kwargs.get('labels'),
        })

        # - returns dummy outputs / loss
        return CausalLMOutputWithPast(
            loss=torch.tensor(0.),
        )

    # mock the from_pretrained function
    old_func = AutoModelForCausalLM.from_pretrained
    def _from_pretrained(model_name_or_path, *args, **kwargs):
        with init_empty_weights():
            model = old_func(
                model_name_or_path, *args, **kwargs,
            )
            model.forward = MethodType(forward, model)
            return model

    return _from_pretrained

def accelerate_prepare(self, model, optimizer, dataloader, scheduler):
    self.state = Mock() # sneak it in

    # skip through the set_epoch if the train loop assumes a distributed
    # dataloader

    def set_epoch(self, *args):
        pass

    dataloader.set_epoch = MethodType(set_epoch, dataloader)
    return model, optimizer, dataloader, scheduler

# - patches

patch_transformers = patch.multiple(
    AutoModelForCausalLM,
    from_pretrained=built_from_pretrained(),
)

patch_accelerate = patch.multiple(
    Accelerator,
    wait_for_everyone=Mock(),
    prepare=accelerate_prepare,
    num_processes=1,
    device=torch.device('cuda'),
    backward=Mock(),
    sync_gradients=False,
)

PATCHES = [
    patch_transformers,
    patch_accelerate,
]

def test_finetune(
    model_name_or_path: str,
    train_file: str,
    max_train_steps: int = 2,
    write_data_to_directory: str = None,
):
    from open_instruct.finetune import main, FlatArguments
    from contextlib import ExitStack

    args = FlatArguments(
        model_name_or_path=model_name_or_path,
        train_file=train_file,
        push_to_hub=False,
        try_launch_beaker_eval_jobs=False,
        output_dir=None,
        max_train_steps=max_train_steps,
    )

    # clear the store
    STORE.clear()
    with ExitStack() as stack:
        for patch in PATCHES:
            stack.enter_context(patch)
        main(args)

    if write_data_to_directory is None:
        return STORE

    os.makedirs(write_data_to_directory)
    for i, data in enumerate(STORE):
        torch.save(data, os.path.join(write_data_to_directory, f'batch_{i}.pt'))

if __name__ == "__main__":
    import fire
    data = fire.Fire(test_finetune)
    