import os.path as osp
import shutil
import warnings
from collections import OrderedDict

import torch
import torch.nn as nn

from .iotools import mkdir_if_missing

from contextlib import suppress
from pathlib import Path


def save_checkpoint(state, save_dir, is_best=False, remove_module_from_keys=False):
    mkdir_if_missing(save_dir)
    if remove_module_from_keys:
        # remove 'module.' in state_dict's keys
        state_dict = state["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            if k.startswith("module."):
                k = k[7:]
            new_state_dict[k] = v
        state["state_dict"] = new_state_dict
    # save
    epoch = state["epoch"]
    fpath = osp.join(save_dir, "model.pth.tar-" + "{:02d}".format(epoch)+".ckpt")
    torch.save(state, fpath)
    print(f'Checkpoint saved to "{fpath}"')
    if is_best:
        shutil.copy(fpath, osp.join(osp.dirname(fpath), "best_model.pth.tar"))

def get_latest_ckpt(path, reverse=False, suffix='.ckpt'):
    """Load latest checkpoint from target directory. Return None if no checkpoints are found."""
    path, file = Path(path), None
    files = (f for f in sorted(path.iterdir(), reverse=not reverse) if f.suffix == suffix)
    with suppress(StopIteration):
        file = next(f for f in files)
    return file

def resume_from_checkpoint(ckpt_dir, model, optimizer=None):
    ckpt_file = get_latest_ckpt(ckpt_dir)
    if ckpt_file is None :
        return 0
    print(f'Loading checkpoint from "{ckpt_file}"')
    ckpt = torch.load(ckpt_file)
    model.load_state_dict(ckpt["state_dict"])
    print("Loaded model weights")
    if optimizer is not None:
        optimizer.load_state_dict(ckpt["optimizer"])
        print("Loaded optimizer")
    start_epoch = ckpt["epoch"]
    print(
        "** previous epoch = {0}\t previous avg loss = {1}".format(
            start_epoch, ckpt["Avg_Train_Loss"]
        )
    )
    return start_epoch