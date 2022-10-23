#!/usr/bin/python3.6

import copy
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

from batch_gen import BatchGenerator
from config import Config, pGTType
from eval import evaluate


class MultiStageModel(nn.Module):
    def __init__(self, num_stages, num_layers, num_f_maps, dim, num_classes):
        super(MultiStageModel, self).__init__()
        self.tower_stage = TowerModel(num_layers, num_f_maps, dim, num_classes)
        self.single_stages = nn.ModuleList(
            [
                copy.deepcopy(
                    SingleStageModel(
                        num_layers, num_f_maps, num_classes, num_classes, 3
                    )
                )
                for _ in range(num_stages - 1)
            ]
        )

    def forward(self, x, mask):
        middle_out, out = self.tower_stage(x, mask)
        middle_out_all = middle_out.unsqueeze(0)
        outputs = out.unsqueeze(0)
        for s in self.single_stages:
            middle_out, out = s(F.softmax(out, dim=1) * mask[:, 0:1, :], mask)
            outputs = torch.cat((outputs, out.unsqueeze(0)), dim=0)
            middle_out_all = torch.cat((middle_out_all, middle_out.unsqueeze(0)), dim=0)
        return middle_out_all, outputs


class TowerModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes):
        super(TowerModel, self).__init__()
        self.stage1 = SingleStageModel(num_layers, num_f_maps, dim, num_classes, 3)
        self.stage2 = SingleStageModel(num_layers, num_f_maps, dim, num_classes, 5)

    def forward(self, x, mask):
        out1, final_out1 = self.stage1(x, mask)
        out2, final_out2 = self.stage2(x, mask)

        return out1 + out2, final_out1 + final_out2


class SingleStageModel(nn.Module):
    def __init__(self, num_layers, num_f_maps, dim, num_classes, kernel_size):
        super(SingleStageModel, self).__init__()
        # noinspection PyTypeChecker
        self.conv_1x1 = nn.Conv1d(dim, num_f_maps, 1)
        self.layers = nn.ModuleList(
            [
                copy.deepcopy(
                    DilatedResidualLayer(2 ** i, num_f_maps, num_f_maps, kernel_size)
                )
                for i in range(num_layers)
            ]
        )
        # noinspection PyTypeChecker
        self.conv_out = nn.Conv1d(num_f_maps, num_classes, 1)

    def forward(self, x, mask):
        out = self.conv_1x1(x)
        for layer in self.layers:
            out = layer(out, mask)
        final_out = self.conv_out(out) * mask[:, 0:1, :]
        return out, final_out


class DilatedResidualLayer(nn.Module):
    def __init__(self, dilation, in_channels, out_channels, kernel_size):
        super(DilatedResidualLayer, self).__init__()
        padding = int(dilation + dilation * (kernel_size - 3) / 2)
        self.conv_dilated = nn.Conv1d(
            in_channels, out_channels, kernel_size, padding=padding, dilation=dilation
        )
        # noinspection PyTypeChecker
        self.conv_1x1 = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout()

    def forward(self, x, mask):
        out = F.relu(self.conv_dilated(x))
        out = self.conv_1x1(out)
        out = self.dropout(out)
        return (x + out) * mask[:, 0:1, :]


class Trainer:
    def __init__(
        self, cfg: Config, num_blocks, num_layers, num_f_maps, dim, num_classes, writer
    ):
        self.cfg = cfg
        self.model = MultiStageModel(
            num_blocks, num_layers, num_f_maps, dim, num_classes
        )
        self.boundary_model = SingleStageModel(1, 32, dim, 2, 1)
        self.ce = nn.CrossEntropyLoss(ignore_index=-100)
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.mse = nn.MSELoss(reduction="none")
        self.num_classes = num_classes
        self.writer = writer

    # noinspection PyMethodMayBeStatic
    def confidence_loss(self, pred, confidence_mask, device):
        batch_size = pred.size(0)
        pred = F.log_softmax(pred, dim=1)
        loss = 0
        for b in range(batch_size):
            num_frame = confidence_mask[b].shape[2]
            # noinspection PyTypeChecker
            m_mask = torch.from_numpy(confidence_mask[b]).type(torch.float).to(device)
            left = pred[b, :, 1:] - pred[b, :, :-1]
            left = torch.clamp(left[:, :num_frame] * m_mask[0], min=0)
            left = torch.sum(left) / torch.sum(m_mask[0])
            loss += left

            right = pred[b, :, :-1] - pred[b, :, 1:]
            right = torch.clamp(right[:, :num_frame] * m_mask[1], min=0)
            right = torch.sum(right) / torch.sum(m_mask[1])
            loss += right

        return loss

    def train(
        self,
        save_dir,
        batch_gen: BatchGenerator,
        num_epochs,
        batch_size,
        learning_rate,
        device,
        results_dir,
        features_path,
        vid_list_file_tst,
        actions_dict,
        sample_rate,
        dataset,
        split,
        notes,
        mem_fs,
    ):
        self.model.train()
        self.model.to(device)
        optimizer = optim.Adam(
            self.model.parameters(), lr=learning_rate, weight_decay=self.cfg.network.wd
        )
        second_stage_start_at = self.cfg.pgt_training_start_at
        s_epoch = 0

        for epoch in range(s_epoch, num_epochs):
            epoch_loss = 0
            correct = 0
            total = 0
            correct_pgt = 0
            while batch_gen.has_next():
                (
                    batch_input,
                    batch_target,
                    mask,
                    batch_confidence,
                    _,
                ) = batch_gen.next_batch(batch_size)
                batch_input, batch_target, mask = (
                    batch_input.to(device),
                    batch_target.to(device),
                    mask.to(device),
                )
                optimizer.zero_grad()
                middle_pred_all, predictions = self.model(batch_input, mask)

                # Generate pseudo labels after training 30 epochs for getting more accurate labels
                if epoch < second_stage_start_at:
                    batch_boundary = batch_gen.get_single_random(
                        batch_size, batch_input.size(-1)
                    ).to(predictions.device)
                else:
                    batch_boundary = batch_gen.canonical_generate_pgt(
                        batch_size,
                        logits=predictions[-1],
                        features=middle_pred_all[-1],
                        pgt_type=self.cfg.pgt_type,
                    )
                    gt_pgt = batch_gen.canonical_generate_pgt(
                        batch_size,
                        logits=predictions[-1],
                        features=middle_pred_all[-1],
                        pgt_type=pGTType.oracle,
                    )
                    correct_pgt += (
                        ((gt_pgt == batch_boundary).float() * mask[:, 0, :].squeeze(1))
                        .sum()
                        .item()
                    )

                loss = torch.tensor(0.0).to(device)
                for p in predictions:
                    loss += self.ce(
                        p.transpose(2, 1).contiguous().view(-1, self.num_classes),
                        batch_boundary.view(-1),
                    )
                    loss += self.cfg.loss.smoothing_factor * torch.mean(
                        torch.clamp(
                            self.mse(
                                F.log_softmax(p[:, :, 1:], dim=1),
                                F.log_softmax(p.detach()[:, :, :-1], dim=1),
                            ),
                            min=0,
                            max=self.cfg.loss.smoothing_clamp_max,
                        )
                        * mask[:, :, 1:]
                    )

                    loss += self.cfg.loss.confidence_factor * self.confidence_loss(
                        p, batch_confidence, device
                    )

                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                _, predicted = torch.max(predictions[-1].data, 1)
                correct += (
                    ((predicted == batch_target).float() * mask[:, 0, :].squeeze(1))
                    .sum()
                    .item()
                )
                total += torch.sum(mask[:, 0, :]).item()

            batch_gen.reset()

            if not self.cfg.no_save_no_writer:
                torch.save(
                    self.model.state_dict(),
                    os.path.join(save_dir, f"epoch-{str(epoch + 1)}.model"),
                )
                torch.save(
                    optimizer.state_dict(),
                    os.path.join(save_dir, f"epoch-{str(epoch + 1)}.opt"),
                )
            if not self.cfg.no_save_no_writer:
                self.writer.add_scalar(
                    "train/Loss",
                    epoch_loss / len(batch_gen.list_of_examples),
                    epoch + 1,
                )
                self.writer.add_scalar("train/Acc", float(correct) / total, epoch + 1)
                self.writer.add_scalar(
                    "train/PGT Acc", float(correct_pgt) / total, epoch + 1
                )
            print(
                "[epoch %d]: epoch loss = %f,   acc = %f"
                % (
                    epoch + 1,
                    epoch_loss / len(batch_gen.list_of_examples),
                    float(correct) / total,
                ),
                flush=True,
            )
            if (epoch + 1) % 10 == 0 or epoch + 1 == num_epochs:
                self.predict(
                    save_dir,
                    results_dir,
                    features_path,
                    vid_list_file_tst,
                    epoch + 1,
                    actions_dict,
                    device,
                    sample_rate,
                    mem_fs,
                )
                acc, edit, f1_all = evaluate(self.cfg, dataset, split, notes, mem_fs)
                if not self.cfg.no_save_no_writer:
                    self.writer.add_scalar("test/Acc", acc, epoch + 1)
                    self.writer.add_scalar("test/Edit", edit, epoch + 1)
                overlap = [0.1, 0.25, 0.5]
                for s in range(len(overlap)):
                    if not self.cfg.no_save_no_writer:
                        self.writer.add_scalar(
                            "test/F1@%0.2f_2D" % overlap[s],
                            f1_all[s],
                            global_step=epoch + 1,
                        )
                self.model.train()
                self.model.to(device)

    def predict(
        self,
        model_dir,
        results_dir,
        features_path,
        vid_list_file,
        epoch,
        actions_dict,
        device,
        sample_rate,
        mem_fs,  # the memory fs
    ):
        self.model.eval()
        with torch.inference_mode():
            self.model.to(device)
            if not self.cfg.no_save_no_writer:
                self.model.load_state_dict(
                    torch.load(os.path.join(model_dir, f"epoch-{str(epoch)}.model"))
                )
            file_ptr = open(vid_list_file, "r")
            list_of_vids = file_ptr.read().split("\n")[:-1]
            file_ptr.close()
            for vid in list_of_vids:
                # print(vid)
                features = np.load(
                    os.path.join(features_path, vid.split(".")[0] + ".npy")
                )
                features = features[:, ::sample_rate]
                input_x = torch.tensor(features, dtype=torch.float)
                input_x.unsqueeze_(0)
                input_x = input_x.to(device)
                _, predictions = self.model(
                    input_x, torch.ones(input_x.size(), device=device)
                )
                _, predicted = torch.max(predictions[-1].data, 1)
                predicted = predicted.squeeze()
                recognition = []
                for i in range(len(predicted)):
                    index = list(actions_dict.values()).index(predicted[i].item())
                    recognition = np.concatenate(
                        (recognition, [list(actions_dict.keys())[index]] * sample_rate)
                    )
                f_name = vid.split("/")[-1].split(".")[0]
                f_ptr = mem_fs.open(os.path.join(results_dir, f_name), "w")
                f_ptr.write("### Frame level recognition: ###\n")
                f_ptr.write(" ".join(recognition))
                f_ptr.close()
