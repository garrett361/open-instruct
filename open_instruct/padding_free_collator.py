import warnings
from dataclasses import dataclass

import torch
from transformers import DefaultDataCollator


@dataclass
class TensorDataCollatorWithFlattening(DefaultDataCollator):
    """
    Data collator used for padding free approach. Does the following:

    - concatate the entire mini batch into single long sequence [1, total_tokens]
    - uses `separator_id` to separate sequences within the concatenated `labels`, default value is -100
    - no padding will be added, returns `input_ids`, `labels` and `position_ids`
    """

    def __init__(
        self,
        *args,
        return_flash_attn_kwargs=True,
        return_position_ids=True,
        separator_id=-100,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.return_flash_attn_kwargs = return_flash_attn_kwargs
        self.return_position_ids = return_position_ids
        self.separator_id = separator_id
        warnings.warn(
            "Using `DataCollatorWithFlattening` will flatten the entire mini batch into single long sequence."
            "Make sure your attention computation is able to handle it!"
        )

    def __call__(self, features, return_tensors=None, separator_id=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if separator_id is None:
            separator_id = self.separator_id
        assert self.return_flash_attn_kwargs, (
            "Only should be used with return_flash_attn_kwargs=True"
        )
        if self.return_flash_attn_kwargs:
            cu_seq_lens = [0]
            max_length = 0
        if self.return_position_ids:
            pos_ids = []
        is_labels_provided = "labels" in features[0]
        ret = {"input_ids": [], "labels": []}
        separator = torch.tensor(
            [separator_id],
            dtype=features[0]["input_ids"].dtype,
            device=features[0]["input_ids"].device,
        )
        for item in features:
            input_ids = item["input_ids"]
            ret["input_ids"].append(input_ids)
            if is_labels_provided:
                ret["labels"].append(separator)
                ret["labels"].append(item["labels"][1:])
            else:
                ret["labels"].append(separator)
                ret["labels"].append(input_ids[1:])
            if self.return_flash_attn_kwargs:
                cu_seq_lens.append(cu_seq_lens[-1] + len(input_ids))
                max_length = max(max_length, len(input_ids))
            if self.return_position_ids:
                pos_ids.append(torch.arange(input_ids.numel(), device=input_ids.device))

        if self.return_flash_attn_kwargs:
            ret["cu_seq_lens_q"] = ret["cu_seq_lens_k"] = torch.tensor(
                cu_seq_lens, dtype=torch.int32, device=features[0]["input_ids"].device
            )
            ret["max_length_q"] = ret["max_length_k"] = max_length
        if self.return_position_ids:
            ret["position_ids"] = torch.cat(pos_ids, dim=0)[None]
        ret["input_ids"] = torch.cat(ret["input_ids"], dim=0)[None]
        ret["labels"] = torch.cat(ret["labels"], dim=0)[None]
        return ret


class DummyLoader:
    def __init__(self, args, accelerator, offset=100) -> None:
        self.args = args
        self.accelerator = accelerator
        self.offset = offset # For avoiding low-numbered special toks

    def __iter__(self):
        tensors = torch.arange(
            self.offset,
            self.args.per_device_train_batch_size * self.args.max_seq_length
            + self.offset,
            device=self.accelerator.device,
        )
        if self.args.padding_free:
            tensors = tensors[None]
            position_ids = torch.cat(
                [
                    torch.arange(
                        self.args.max_seq_length,
                        device=self.accelerator.device,
                    )[None]
                    for n in range(self.args.per_device_train_batch_size)
                ],
                dim=-1,
            )
            max_seq_length = self.args.max_seq_length
            cu_seq_lens = torch.tensor(
                [0, self.args.max_seq_length], device=tensors.device, dtype=torch.int32
            )
            batch = {
                "input_ids": tensors,
                "labels": tensors,
                "position_ids": position_ids,
                "max_length_q": max_seq_length,
                "max_length_k": max_seq_length,
                "cu_seq_lens_q": cu_seq_lens,
                "cu_seq_lens_k": cu_seq_lens,
            }
        else:
            tensors = tensors.reshape(
                self.args.per_device_train_batch_size, self.args.max_seq_length
            )
            batch = {
                "input_ids": tensors,
                "labels": tensors,
                "attention_mask": torch.ones_like(tensors),
            }
        while True:
            yield batch

    def __len__(self):
        return 1000

    def set_epoch(self, *args, **kwargs):
        return
