"""
Copyright (c) Meta Platforms, Inc. and affiliates.
All rights reserved.
This source code is licensed under the license found in the
LICENSE file in the root directory of this source tree.
"""

import os
from typing import Dict, Iterable, List, Union

import pandas as pd

import numpy as np
import torch
from torch.utils import data
import librosa

from utils.misc import prGreen


class Social(data.Dataset):
    def __init__(
        self,
        args,
        data_dict: Dict[str, Iterable],
        split: str = "train",
        chunk: bool = False,
        add_padding: bool = True,
    ) -> None:
        if args.data_format == "face":
            prGreen("[dataset.py] training face only model")
            data_dict["data"] = data_dict["face"]
        elif args.data_format == "pose":
            prGreen("[dataset.py] training pose only model")
            missing = []
            for d in data_dict["data"]:
                missing.append(np.ones_like(d))
            data_dict["missing"] = missing

        # set up variables for dataloader
        self.data_format = args.data_format
        self.add_frame_cond = args.add_frame_cond
        self._register_keyframe_step()
        self.data_root = args.data_root
        self.max_seq_length = args.max_seq_length
        if hasattr(args, "curr_seq_length") and args.curr_seq_length is not None:
            self.max_seq_length = args.curr_seq_length
        prGreen([f"[dataset.py] sequences of {self.max_seq_length}"])
        self.add_padding = add_padding
        self.audio_per_frame = 1600
        self.max_audio_length = self.max_seq_length * self.audio_per_frame
        self.min_seq_length = 400

        # set up training/validation splits
        train_idx = list(range(0, len(data_dict["data"]) - 6))
        val_idx = list(range(len(data_dict["data"]) - 6, len(data_dict["data"]) - 4))
        test_idx = list(range(len(data_dict["data"]) - 4, len(data_dict["data"])))
        self.split = split
        if split == "train":
            self._pick_sequences(data_dict, train_idx)
        elif split == "val":
            self._pick_sequences(data_dict, val_idx)
        else:
            self._pick_sequences(data_dict, test_idx)
        self.chunk = chunk
        if split == "test":
            print("[dataset.py] chunking data...")
            self._chunk_data()
        self._load_std()
        prGreen(
            f"[dataset.py] {split} | {len(self.data)} sequences ({self.data[0].shape}) | total len {self.total_len}"
        )

    def inv_transform(
        self, data: Union[np.ndarray, torch.Tensor], data_type: str
    ) -> Union[np.ndarray, torch.Tensor]:
        if data_type == "pose":
            std = self.std
            mean = self.mean
        elif data_type == "face":
            std = self.face_std
            mean = self.face_mean
        elif data_type == "audio":
            std = self.audio_std
            mean = self.audio_mean
        else:
            assert False, f"datatype not defined: {data_type}"

        if torch.is_tensor(data):
            return data * torch.tensor(
                std, device=data.device, requires_grad=False
            ) + torch.tensor(mean, device=data.device, requires_grad=False)
        else:
            return data * std + mean

    def _pick_sequences(self, data_dict: Dict[str, Iterable], idx: List[int]) -> None:
        self.data = np.take(data_dict["data"], idx, axis=0)
        self.missing = np.take(data_dict["missing"], idx, axis=0)
        self.audio = np.take(data_dict["audio"], idx, axis=0)
        self.lengths = np.take(data_dict["lengths"], idx, axis=0)
        self.total_len = sum([len(d) for d in self.data])

    def _load_std(self) -> None:
        stats = torch.load(os.path.join(self.data_root, "data_stats.pth"))
        print(
            f'[dataset.py] loading from... {os.path.join(self.data_root, "data_stats.pth")}'
        )
        self.mean = stats["pose_mean"].reshape(-1)
        self.std = stats["pose_std"].reshape(-1)
        self.face_mean = stats["code_mean"]
        self.face_std = stats["code_std"]
        self.audio_mean = stats["audio_mean"]
        self.audio_std = stats["audio_std_flat"]

    def _chunk_data(self) -> None:
        chunk_data = []
        chunk_missing = []
        chunk_lengths = []
        chunk_audio = []
        # create sequences of set lengths
        for d_idx in range(len(self.data)):
            curr_data = self.data[d_idx]
            curr_missing = self.missing[d_idx]
            curr_audio = self.audio[d_idx]
            end_range = len(self.data[d_idx]) - self.max_seq_length
            for chunk_idx in range(0, end_range, self.max_seq_length):
                chunk_end = chunk_idx + self.max_seq_length
                curr_data_chunk = curr_data[chunk_idx:chunk_end, :]
                curr_missing_chunk = curr_missing[chunk_idx:chunk_end, :]
                curr_audio_chunk = curr_audio[
                    chunk_idx * self.audio_per_frame : chunk_end * self.audio_per_frame,
                    :,
                ]
                if curr_data_chunk.shape[0] < self.max_seq_length:
                    # do not add a short chunk to the list
                    continue
                chunk_lengths.append(curr_data_chunk.shape[0])
                chunk_data.append(curr_data_chunk)
                chunk_missing.append(curr_missing_chunk)
                chunk_audio.append(curr_audio_chunk)
        idx = np.random.permutation(len(chunk_data))
        print("==> shuffle", idx)
        self.data = np.take(chunk_data, idx, axis=0)
        self.missing = np.take(chunk_missing, idx, axis=0)
        self.lengths = np.take(chunk_lengths, idx, axis=0)
        self.audio = np.take(chunk_audio, idx, axis=0)
        self.total_len = len(self.data)

    def _register_keyframe_step(self) -> None:
        if self.add_frame_cond == 1:
            self.step = 30
        if self.add_frame_cond is None:
            self.step = 1

    def _pad_sequence(
        self, sequence: np.ndarray, actual_length: int, max_length: int
    ) -> np.ndarray:
        sequence = np.concatenate(
            (
                sequence,
                np.zeros((max_length - actual_length, sequence.shape[-1])),
            ),
            axis=0,
        )
        return sequence

    def _get_idx(self, item: int) -> int:
        cumulative_len = 0
        seq_idx = 0
        while item > cumulative_len:
            cumulative_len += len(self.data[seq_idx])
            seq_idx += 1
        item = seq_idx - 1
        return item

    def _get_random_subsection(
        self, data_dict: Dict[str, Iterable]
    ) -> Dict[str, np.ndarray]:
        isnonzero = False
        while not isnonzero:
            start = np.random.randint(0, data_dict["m_length"] - self.max_seq_length)
            if self.add_padding:
                length = (
                    np.random.randint(self.min_seq_length, self.max_seq_length)
                    if not self.split == "test"
                    else self.max_seq_length
                )
            else:
                length = self.max_seq_length
            curr_missing = data_dict["missing"][start : start + length]
            isnonzero = np.any(curr_missing)
        missing = curr_missing
        motion = data_dict["motion"][start : start + length, :]
        keyframes = motion[:: self.step]
        audio = data_dict["audio"][
            start * self.audio_per_frame : (start + length) * self.audio_per_frame,
            :,
        ]
        data_dict["m_length"] = len(motion)
        data_dict["k_length"] = len(keyframes)
        data_dict["a_length"] = len(audio)

        if data_dict["m_length"] < self.max_seq_length:
            motion = self._pad_sequence(
                motion, data_dict["m_length"], self.max_seq_length
            )
            missing = self._pad_sequence(
                missing, data_dict["m_length"], self.max_seq_length
            )
            audio = self._pad_sequence(
                audio, data_dict["a_length"], self.max_audio_length
            )
            max_step_length = len(np.zeros(self.max_seq_length)[:: self.step])
            keyframes = self._pad_sequence(
                keyframes, data_dict["k_length"], max_step_length
            )
        data_dict["motion"] = motion
        data_dict["keyframes"] = keyframes
        data_dict["audio"] = audio
        data_dict["missing"] = missing
        return data_dict

    def __len__(self) -> int:
        return self.total_len

    def __getitem__(self, item: int) -> Dict[str, np.ndarray]:
        # figure out which sequence to randomly sample from
        if not self.split == "test":
            item = self._get_idx(item)
        motion = self.data[item]
        audio = self.audio[item]
        m_length = self.lengths[item]
        missing = self.missing[item]
        a_length = len(audio)
        # Z Normalization
        if self.data_format == "pose":
            motion = (motion - self.mean) / self.std
        elif self.data_format == "face":
            motion = (motion - self.face_mean) / self.face_std
        audio = (audio - self.audio_mean) / self.audio_std
        keyframes = motion[:: self.step]
        k_length = len(keyframes)
        data_dict = {
            "motion": motion,
            "m_length": m_length,
            "audio": audio,
            "a_length": a_length,
            "keyframes": keyframes,
            "k_length": k_length,
            "missing": missing,
        }
        if not self.split == "test" and not self.chunk:
            data_dict = self._get_random_subsection(data_dict)
        if self.data_format == "face":
            data_dict["motion"] *= data_dict["missing"]
        return data_dict



class MEADDataset(data.Dataset):
    def __init__(self, args, split):
        super().__init__()
        self.subject_dict = self.get_subject_dict()
        self.dataset_root = '/ytech_data_ssd/ytech_vhuman_ssd/zhanghaoxian/processed/MEAD'
        self.df = self.load_df(split)
        self.data_format = args.data_format

        self.split = split
        self.max_seq_length = args.max_seq_length       # 600
        self.emo_list = ['angry', 'contempt', 'disgusted', 'fear', 'happy', 'sad', 'surprised', 'neutral']


    def __getitem__(self, index):
        row = self.df.iloc[index]  # [pid, emotion, intensity, flame_id, audio_id]
        audio_path = os.path.join(self.dataset_root, 'audio', row['pid'], 'audio', row['emotion'], row['intensity'], row['audio_id'])
        blenshape_path = os.path.join(self.dataset_root, 'blendshape',
                                    row['pid']+"-"+row['emotion']+"-"+row['intensity']+"-"+row['flame_id'].replace('.npz', '_mp.npy'))
        audio, audio_length = self.get_audio(audio_path)
        blendshape, blendshape_length = self.get_blendshape(blenshape_path)

        
        true_audio_length = int(blendshape_length * 3 * 1600 / 2.5)
        up_pad = 1600 - (audio_length % 1600)

        # print(f'old {audio_length} -> new {audio_length + up_pad}')
        audio = torch.concat([audio, torch.zeros_like(audio)[:up_pad]])
        audio_length = torch.tensor(audio.shape[0])



        emotion = torch.from_numpy(np.eye(8)[self.emo_list.index(row['emotion'])])

        intensity = torch.tensor(int(row['intensity'][-1]))

        missing = torch.ones_like(blendshape)  # currently we don't manage the missing on MEAD dataset

        # print('blendshape shape', blendshape_length, 'audio shape', audio_length)

        data_dict = {
            "motion": blendshape,
            "m_length": blendshape_length,
            "audio": audio,
            "a_length": audio_length,
            "missing": missing,
            'pid': self.get_id_one_hot(row['pid']),
            'emotion': emotion, 
            'intensity': intensity
        }

        # TODO: augmentation, after the pipeline is ready to train
        # if not self.split == "test":
        #     data_dict = self._get_random_subsection(data_dict)
        if self.data_format == "arkit":
            data_dict["motion"] *= data_dict["missing"]
        return data_dict


    @staticmethod
    def get_audio(audio_path: str):
        audio, _ = librosa.load(audio_path, sr=48000, mono=True)
        audio = torch.from_numpy(audio)
        length = torch.tensor(audio.shape[0])
        return audio, length

    @staticmethod
    def get_blendshape(flame_path: str):
        data = np.load(flame_path, allow_pickle=True)
        return torch.from_numpy(data).to(torch.float32).squeeze(0), torch.tensor(data.shape[0])     # (T, 51), T

    def __len__(self):
        return len(self.df)


    def _get_random_subsection(
        self, data_dict: Dict[str, Iterable]
    ) -> Dict[str, np.ndarray]:
        isnonzero = False
        while not isnonzero:
            start = np.random.randint(0, data_dict["m_length"] - self.max_seq_length)
            if self.add_padding:
                length = (
                    np.random.randint(self.min_seq_length, self.max_seq_length)
                    if not self.split == "test"
                    else self.max_seq_length
                )
            else:
                length = self.max_seq_length
            curr_missing = data_dict["missing"][start : start + length]
            isnonzero = np.any(curr_missing)
        missing = curr_missing
        motion = data_dict["motion"][start : start + length, :]
        keyframes = motion[:: self.step]
        audio = data_dict["audio"][
            start * self.audio_per_frame : (start + length) * self.audio_per_frame,
            :,
        ]
        data_dict["m_length"] = len(motion)
        data_dict["k_length"] = len(keyframes)
        data_dict["a_length"] = len(audio)

        if data_dict["m_length"] < self.max_seq_length:
            motion = self._pad_sequence(
                motion, data_dict["m_length"], self.max_seq_length
            )
            missing = self._pad_sequence(
                missing, data_dict["m_length"], self.max_seq_length
            )
            audio = self._pad_sequence(
                audio, data_dict["a_length"], self.max_audio_length
            )
            max_step_length = len(np.zeros(self.max_seq_length)[:: self.step])
            keyframes = self._pad_sequence(
                keyframes, data_dict["k_length"], max_step_length
            )
        data_dict["motion"] = motion
        data_dict["keyframes"] = keyframes
        data_dict["audio"] = audio
        data_dict["missing"] = missing
        return data_dict

    def load_df(self, split):
        df = pd.read_csv(os.path.join(self.dataset_root, 'mead.csv'))
        return df[df['pid'].isin(self.subject_dict[split])]


    def get_subject_dict(self):
        val_set = ['M041', 'W038']
        test_set = ['M042', 'W040']
        train_set = [x for x in os.listdir('/ytech_data_ssd/ytech_vhuman_ssd/zhanghaoxian/processed/MEAD/audio') if x not in test_set and x not in val_set]
        return {
            'train': train_set,
            'val': val_set,
            'test': test_set
        }
    
    def get_id_one_hot(self, pid:str):
        if pid in self.subject_dict['train']:
            return torch.from_numpy(np.eye(len(self.subject_dict['train']))[self.subject_dict['train'].index(pid)])
        else:
            return torch.from_numpy(np.eye(len(self.subject_dict['train'])))
    
    def inv_transform(data, data_type):
        return data

