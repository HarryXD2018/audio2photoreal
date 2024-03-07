"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import torch
from torch.utils.data._utils.collate import default_collate


def lengths_to_mask(lengths, max_len):
    mask = torch.arange(max_len, device=lengths.device).expand(
        len(lengths), max_len
    ) < lengths.unsqueeze(1)
    return mask


def collate_tensors(batch):
    dims = batch[0].dim()
    max_size = [max([b.size(i) for b in batch]) for i in range(dims)]
    size = (len(batch),) + tuple(max_size)
    canvas = batch[0].new_zeros(size=size)
    for i, b in enumerate(batch):
        sub_tensor = canvas[i]
        for d in range(dims):
            sub_tensor = sub_tensor.narrow(d, 0, b.size(d))
        sub_tensor.add_(b)
    return canvas


## social collate
def collate_v2(batch):
    notnone_batches = [b for b in batch if b is not None]
    databatch = [b["inp"] for b in notnone_batches]
    missingbatch = [b["missing"] for b in notnone_batches]
    audiobatch = [b["audio"] for b in notnone_batches]
    lenbatch = [b["lengths"] for b in notnone_batches]
    alenbatch = [b["audio_lengths"] for b in notnone_batches]
    keyframebatch = [b["keyframes"] for b in notnone_batches]
    klenbatch = [b["key_lengths"] for b in notnone_batches]

    databatchTensor = collate_tensors(databatch)
    missingbatchTensor = collate_tensors(missingbatch)
    audiobatchTensor = collate_tensors(audiobatch)
    lenbatchTensor = torch.as_tensor(lenbatch)
    alenbatchTensor = torch.as_tensor(alenbatch)
    keyframeTensor = collate_tensors(keyframebatch)
    klenbatchTensor = torch.as_tensor(klenbatch)

    maskbatchTensor = (
        lengths_to_mask(lenbatchTensor, databatchTensor.shape[-1])
        .unsqueeze(1)
        .unsqueeze(1)
    )  # unqueeze for broadcasting
    motion = databatchTensor
    cond = {
        "y": {
            "missing": missingbatchTensor,
            "mask": maskbatchTensor,
            "lengths": lenbatchTensor,
            "audio": audiobatchTensor,
            "alengths": alenbatchTensor,
            "keyframes": keyframeTensor,
            "klengths": klenbatchTensor,
        }
    }
    return motion, cond


def social_collate(batch):
    adapted_batch = [
        {
            "inp": torch.tensor(b["motion"].T).to(torch.float32).unsqueeze(1),
            "lengths": b["m_length"],
            "audio": b["audio"]
            if torch.is_tensor(b["audio"])
            else torch.tensor(b["audio"]).to(torch.float32),
            "keyframes": torch.tensor(b["keyframes"]).to(torch.float32),
            "key_lengths": b["k_length"],
            "audio_lengths": b["a_length"],
            "missing": torch.tensor(b["missing"]).to(torch.float32),
        }
        for b in batch
    ]
    return collate_v2(adapted_batch)


def mead_collate(batch):
    return PadSequence()(batch)

class PadSequence:
    def __init__(self):
        self.need_pad = ['audio', 'motion', 'missing', 'clip_seq']

    def __call__(self, batch):
        results = {}
        for key in batch[0].keys():
            if key in self.need_pad:
                results[key] = torch.nn.utils.rnn.pad_sequence([x[key] for x in batch], batch_first=True)
            else:
                results[key] = torch.stack([x[key] for x in batch])
        cond = {
            'y': {
                k: results[k] for k in results.keys() if k != 'motion' 
            }
        }
        maskbatchTensor = (
            lengths_to_mask(results['m_length'], results['motion'].shape[-1])
            .unsqueeze(1)
            .unsqueeze(1)
        )  # unqueeze for broadcasting
        
        cond['y']['mask'] = maskbatchTensor
        motion = results['motion']
        return motion, cond
