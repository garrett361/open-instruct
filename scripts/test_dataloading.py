import torch
import os

from unittest.mock import patch, Mock
from accelerate import Accelerator


from transformers import AutoModelForCausalLM
from accelerate import init_empty_weights
from transformers.modeling_outputs import CausalLMOutputWithPast

from typing import Callable, List
from types import MethodType

# builds the from_pretrained to pass through AutoModelForCausalLM
# - loads the model on the meta device

def built_from_pretrained(
    store: List,
    extra_keys: List = ['labels'],
):

    def forward(self, input_ids, *args, **kwargs):
        # - hook to capture the input_ids
        store.append(
            {
                'input_ids': input_ids,
                **{k:kwargs.get(k) for k in extra_keys},
            }
        )

        # - returns dummy outputs / loss
        return CausalLMOutputWithPast(
            loss=torch.tensor(0.),
            logits=torch.zeros(input_ids.shape + (self.config.vocab_size,))
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

patch_accelerate = patch.multiple(
    Accelerator,
    wait_for_everyone=Mock(),
    prepare=accelerate_prepare,
    num_processes=1,
    device=torch.device('cuda'),
    backward=Mock(),
    sync_gradients=False,
)


def test_tuning_script(
    script: Callable,
    args: object, 
    patches: List[object],
    store: List,
    write_data_to_directory: str = None,
):
    from contextlib import ExitStack

    # clear the store
    store.clear()
    with ExitStack() as stack:
        for patch in patches:
            stack.enter_context(patch)
        script(args)

    if write_data_to_directory is None:
        return store

    os.makedirs(write_data_to_directory)
    for i, data in enumerate(store):
        torch.save(data, os.path.join(write_data_to_directory, f'batch_{i}.pt'))


def test_finetune(
    model_name_or_path: str,
    dataset_name: str = None,
    train_file: str = None,
    max_train_steps: int = 2,
    write_data_to_directory: str = None,
):
    from open_instruct.finetune import main, FlatArguments

    args = FlatArguments(
        model_name_or_path=model_name_or_path,
        dataset_name=dataset_name,
        train_file=train_file,
        push_to_hub=False,
        try_launch_beaker_eval_jobs=False,
        output_dir=None,
        max_train_steps=max_train_steps,
    )

    # - initialize store in transformers patcher
    STORE = []
    patch_transformers = patch.multiple(
        AutoModelForCausalLM,
        from_pretrained=built_from_pretrained(
            STORE
        ),
    )

    return test_tuning_script(
        main, args,
        [
            patch_transformers,
            patch_accelerate,
        ],
        STORE,
        write_data_to_directory,
    )

def test_dpo_tune(
    model_name_or_path: str,
    dataset_name: str = None,
    train_file: str = None,
    max_train_steps: int = 2,
    write_data_to_directory: str = None,
):
    from open_instruct.dpo_tune_cache import main, FlatArguments

    args = FlatArguments(
        model_name_or_path=model_name_or_path,
        dataset_name=dataset_name,
        train_file=train_file,
        push_to_hub=False,
        try_launch_beaker_eval_jobs=False,
        try_auto_save_to_beaker=False,
        output_dir=None,
        max_train_steps=max_train_steps,
    )

    # - initialize store in transformers patcher
    STORE = []
    patch_transformers = patch.multiple(
        AutoModelForCausalLM,
        from_pretrained=built_from_pretrained(
            STORE,
            extra_keys=[],
        ),
    )

    return test_tuning_script(
        main, args,
        [
            patch_transformers,
            patch_accelerate,
        ],
        STORE,
        write_data_to_directory,
    )


if __name__ == "__main__":
    import fire
    data = fire.Fire(test_finetune)
    