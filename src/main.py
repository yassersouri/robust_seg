"""
python src/main.py notes=XXX dataset=gtea split=1
"""

import os
import random

import fs
import hydra
import numpy as np
import torch
from hydra.core.config_store import ConfigStore
from torch.utils.tensorboard import SummaryWriter

from batch_gen import BatchGenerator
from config import Config
from eval import evaluate
from model import Trainer

from datetime import datetime as dt

CONFIG_NAME = "config"

cs = ConfigStore.instance()
cs.store(name=CONFIG_NAME, node=Config)


@hydra.main(config_path=None, config_name=CONFIG_NAME)
def main(cfg: Config):
    device = torch.device(cfg.device)
    set_seed(cfg, fast=True)
    mem_fs = fs.open_fs("mem://")

    # figuring out the annotation and trained_model directory
    cfg.annotation_file = os.path.join(
        cfg.path.annotation_root,
        f"timestamps_{cfg.timestamp_percentage}",
        f"{cfg.dataset}_annotation_all.npy",
    )

    if cfg.no_save_no_writer:
        writer = None
    else:
        writer = SummaryWriter(
            log_dir=os.path.join(
                cfg.path.logs_root, cfg.dataset, f"split-{cfg.split}_notes_{cfg.notes}"
            )
        )

    # use the full temporal resolution @ 15fps
    sample_rate = 1
    # sample input features @ 15fps instead of 30 fps
    # for 50salads, and up-sample the output to 30 fps
    if cfg.dataset == "50salads":
        sample_rate = 2

    vid_list_file = os.path.join(
        cfg.path.data_root,
        cfg.dataset,
        f"splits/train.split{cfg.split}.bundle",
    )
    vid_list_file_tst = os.path.join(
        cfg.path.data_root,
        cfg.dataset,
        f"splits/test.split{cfg.split}.bundle",
    )
    features_path = os.path.join(cfg.path.data_root, cfg.dataset, "features")
    gt_path = os.path.join(cfg.path.data_root, cfg.dataset, "groundTruth")

    mapping_file = os.path.join(cfg.path.data_root, cfg.dataset, "mapping.txt")

    model_dir = os.path.join(
        cfg.path.models_root, cfg.dataset, f"split_{cfg.split}_notes_{cfg.notes}"
    )
    results_dir = os.path.join(
        cfg.path.results_root, cfg.dataset, f"split_{cfg.split}_notes_{cfg.notes}"
    )

    if not cfg.no_save_no_writer:
        os.makedirs(model_dir, exist_ok=True, mode=0o777)

    # results will be created on the memory_fs
    mem_fs.makedirs(results_dir)

    print(
        "{} dataset {} in split {} for single stamp supervision".format(
            cfg.action, cfg.dataset, cfg.split
        ),
        flush=True,
    )
    print(
        "batch size is {}, number of stages is {}, sample rate is {}\n".format(
            cfg.network.bz, cfg.network.num_stages, sample_rate
        ),
        flush=True,
    )

    with open(mapping_file, "r") as file_ptr:
        actions = file_ptr.read().split("\n")[:-1]
    actions_dict = dict()
    for a in actions:
        actions_dict[a.split()[1]] = int(a.split()[0])

    num_classes = len(actions_dict)
    trainer = Trainer(
        cfg,
        cfg.network.num_stages,
        cfg.network.num_layers,
        cfg.network.num_f_maps,
        cfg.network.features_dim,
        num_classes,
        writer,
    )

    if cfg.action == "train":
        batch_gen = BatchGenerator(
            cfg,
            num_classes,
            actions_dict,
            gt_path,
            features_path,
            sample_rate,
            cfg.annotation_file,
        )
        batch_gen.read_data(vid_list_file)

        # Train the model
        trainer.train(
            model_dir,
            batch_gen,
            cfg.network.num_epochs,
            cfg.network.bz,
            cfg.network.lr,
            device,
            results_dir,
            features_path,
            vid_list_file_tst,
            actions_dict,
            sample_rate,
            cfg.dataset,
            cfg.split,
            cfg.notes,
            mem_fs,
        )

    # Predict the output label for each frame in evaluation and output them
    trainer.predict(
        model_dir,
        results_dir,
        features_path,
        vid_list_file_tst,
        cfg.network.num_epochs,
        actions_dict,
        device,
        sample_rate,
        mem_fs,
    )
    # Read output files and measure metrics (F1@10, 25, 50, Edit, Acc)
    acc, edit, f1_all = evaluate(cfg, cfg.dataset, cfg.split, cfg.notes, mem_fs)

    print("\n***** RESULTS *****\n")
    print("Acc: %.4f" % acc)
    if not cfg.no_save_no_writer:
        writer.add_text("Acc", "Acc: %.4f" % acc)
    print("Edit: %.4f" % edit)
    if not cfg.no_save_no_writer:
        writer.add_text("Edit", "Edit: %.4f" % edit)
    overlap = [0.1, 0.25, 0.5]
    for s in range(len(overlap)):
        print("F1@%0.2f: %.4f" % (overlap[s], f1_all[s]))
        if not cfg.no_save_no_writer:
            writer.add_text(
                "F1@%0.2f:" % overlap[s], "F1@%0.2f: %.4f" % (overlap[s], f1_all[s])
            )
    if not cfg.no_save_no_writer:
        writer.close()

    return [f1_all[0], f1_all[1], f1_all[2], edit, acc]


def set_seed(cfg, fast: bool = False):
    seed = cfg.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    if not fast:
        # noinspection PyUnresolvedReferences
        torch.backends.cudnn.deterministic = True


if __name__ == "__main__":
    tic = dt.now()
    main()
    print("Total Time: {}".format(dt.now() - tic))
