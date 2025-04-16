import warnings
from dataclasses import dataclass

import torch
from transformers import DefaultDataCollator
from transformers.data.data_collator import default_data_collator


@dataclass
class TensorDataCollatorWithFlattening(DefaultDataCollator):
    """
    Data collator used for padding free approach. Does the following:

    - concatate the entire mini batch into single long sequence [1, total_tokens]
    - uses `separator_id` to separate sequences within the concatenated `labels`, default value is -100
    - no padding will be added, returns `input_ids`, `labels` and `position_ids`
    """

    def __init__(self, *args, return_position_ids=True, separator_id=-100, **kwargs):
        super().__init__(*args, **kwargs)
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
        is_labels_provided = "labels" in features[0]
        ret = {"input_ids": [], "labels": []}
        if self.return_position_ids:
            ret.update({"position_ids": []})
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
            if self.return_position_ids:
                ret["position_ids"].append(
                    torch.arange(
                        input_ids.numel(),
                        dtype=input_ids.dtype,
                        device=input_ids.device,
                    )
                )
        ret = {k: torch.cat(v, dim=0) for k, v in ret.items()}
        return default_data_collator([ret], return_tensors)
