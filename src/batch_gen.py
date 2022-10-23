#!/usr/bin/python3.6
import datetime
import os
import random

import numpy as np
import torch
import torch.nn.functional as F

from config import Config, pGTType
from pgt import (
    generate_baseline_pgt,
    generate_oracle_pgt,
    generate_pgt_hard,
)


class BatchGenerator(object):
    def __init__(
        self,
        cfg: Config,
        num_classes,
        actions_dict,
        gt_path,
        features_path,
        sample_rate,
        annotation_file,
        device="cpu",
    ):
        self.cfg = cfg
        self.list_of_examples = list()
        self.index = 0
        self.num_classes = num_classes
        self.actions_dict = actions_dict
        self.gt_path = gt_path
        self.features_path = features_path
        self.sample_rate = sample_rate
        self.gt = {}
        self.confidence_mask = {}
        self.device = device

        self.random_index = np.load(annotation_file, allow_pickle=True).item()

    def reset(self):
        self.index = 0
        random.shuffle(self.list_of_examples)

    def has_next(self):
        if self.index < len(self.list_of_examples):
            return True
        return False

    def read_data(self, vid_list_file):
        file_ptr = open(vid_list_file, "r")
        self.list_of_examples = file_ptr.read().split("\n")[:-1]
        file_ptr.close()
        random.shuffle(self.list_of_examples)
        self.generate_confidence_mask()

    def generate_confidence_mask(self):
        for vid in self.list_of_examples:
            file_ptr = open(os.path.join(self.gt_path, vid), "r")
            content = file_ptr.read().split("\n")[:-1]
            classes = np.zeros(len(content), dtype=np.long)
            for i in range(len(classes)):
                classes[i] = self.actions_dict[content[i]]
            classes = classes[:: self.sample_rate]
            self.gt[vid] = classes
            num_frames = classes.shape[0]

            # noinspection PyUnresolvedReferences
            random_idx = self.random_index[vid]

            # Generate mask for confidence loss. There are two masks for both side of timestamps
            left_mask = np.zeros([self.num_classes, num_frames - 1])
            right_mask = np.zeros([self.num_classes, num_frames - 1])
            for j in range(len(random_idx) - 1):
                left_mask[
                    int(classes[random_idx[j]]), random_idx[j] : random_idx[j + 1]
                ] = 1
                right_mask[
                    int(classes[random_idx[j + 1]]), random_idx[j] : random_idx[j + 1]
                ] = 1

            self.confidence_mask[vid] = np.array([left_mask, right_mask])

    def next_batch(self, batch_size):
        batch = self.list_of_examples[self.index : self.index + batch_size]
        self.index += batch_size

        batch_input = []
        batch_target = []
        batch_confidence = []
        batch_names = []
        for vid in batch:
            features = np.load(
                os.path.join(self.features_path, vid.split(".")[0] + ".npy")
            )
            batch_input.append(features[:, :: self.sample_rate])
            batch_target.append(self.gt[vid])
            batch_confidence.append(self.confidence_mask[vid])
            batch_names.append(vid.split(".")[0])

        length_of_sequences = list(map(len, batch_target))
        batch_input_tensor = torch.zeros(
            len(batch_input),
            np.shape(batch_input[0])[0],
            max(length_of_sequences),
            dtype=torch.float,
        )
        batch_target_tensor = torch.ones(
            len(batch_input), max(length_of_sequences), dtype=torch.long
        ) * (-100)
        mask = torch.zeros(
            len(batch_input),
            self.num_classes,
            max(length_of_sequences),
            dtype=torch.float,
        )
        for i in range(len(batch_input)):
            batch_input_tensor[i, :, : np.shape(batch_input[i])[1]] = torch.from_numpy(
                batch_input[i]
            )
            batch_target_tensor[i, : np.shape(batch_target[i])[0]] = torch.from_numpy(
                batch_target[i]
            )
            mask[i, :, : np.shape(batch_target[i])[0]] = torch.ones(
                self.num_classes, np.shape(batch_target[i])[0]
            )

        return (
            batch_input_tensor,
            batch_target_tensor,
            mask,
            batch_confidence,
            batch_names,
        )

    def get_single_random(self, batch_size, max_frames):
        # Generate target for only timestamps. Do not generate pseudo labels at first 30 epochs.
        batch = self.list_of_examples[self.index - batch_size : self.index]
        boundary_target_tensor = torch.ones(
            len(batch), max_frames, dtype=torch.long
        ) * (-100)
        for b, vid in enumerate(batch):
            # noinspection PyUnresolvedReferences
            single_frame = self.random_index[vid]
            gt = self.gt[vid]
            frame_idx_tensor = torch.from_numpy(np.array(single_frame))
            gt_tensor = torch.from_numpy(gt.astype(int))
            boundary_target_tensor[b, frame_idx_tensor] = gt_tensor[frame_idx_tensor]

        return boundary_target_tensor

    def canonical_generate_pgt(
        self, batch_size, logits, features, pgt_type: pGTType = pGTType.hard
    ):
        # This function is to generate pseudo labels
        batch, target_tensor = self.create_output_placeholder(batch_size, logits)
        tic = datetime.datetime.now()
        for b, vid in enumerate(batch):
            # noinspection PyUnresolvedReferences
            timestamps = torch.tensor(self.random_index[vid])
            vid_gt = self.gt[vid]
            timestamp_labels = torch.tensor(vid_gt[timestamps]).to(logits.device)
            nlp = -1 * F.log_softmax(logits[b][:, : vid_gt.shape[0]], dim=0).detach()
            timestamps = timestamps.to(logits.device)

            the_pgt_config = self.cfg.pgt_config

            # this part should change based on pgt type.
            if pgt_type == pGTType.hard:
                target = generate_pgt_hard(
                    nlp,
                    timestamps,
                    timestamp_labels,
                    config=the_pgt_config,
                ).pgt
            elif pgt_type == pGTType.oracle:
                target = generate_oracle_pgt(
                    timestamps,
                    vid_gt,
                    ignore_label=-100,
                ).to(logits.device)
            elif self.cfg.pgt_type == pGTType.baseline:
                target = generate_baseline_pgt(
                    timestamps=timestamps,
                    labels=vid_gt,
                    intermediate_features=features[b],
                    ignore_label=-100,
                )
            else:
                raise NotImplementedError("not supported pGTType.")

            target_tensor[b, : vid_gt.shape[0]] = target
        print("Time to generate pGT {}: ".format(pgt_type), datetime.datetime.now() - tic)
        return target_tensor.to(logits.device)

    def create_output_placeholder(self, batch_size, logits):
        batch = self.list_of_examples[self.index - batch_size : self.index]
        num_video, _, max_frames = logits.size()
        target_tensor = torch.ones(num_video, max_frames, dtype=torch.long) * (-100)
        return batch, target_tensor
