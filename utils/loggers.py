# Copyright (c) EEEM071, University of Surrey

import os
import os.path as osp
import sys

from .iotools import mkdir_if_missing


class Logger:
    """
    Write console output to external text file.
    Code imported from https://github.com/Cysu/open-reid/blob/master/reid/utils/logging.py.
    """

    def __init__(self, fpath=None):
        self.console = sys.stdout
        self.file = None
        if fpath is not None:
            mkdir_if_missing(osp.dirname(fpath))
            self.file = open(fpath, "a")

    def __del__(self):
        self.close()

    def __enter__(self):
        pass

    def __exit__(self, *args):
        self.close()

    def write(self, msg):
        self.console.write(msg)
        if self.file is not None:
            self.file.write(msg)

    def flush(self):
        self.console.flush()
        if self.file is not None:
            self.file.flush()
            os.fsync(self.file.fileno())

    def close(self):
        self.console.close()
        if self.file is not None:
            self.file.close()


class RankLogger:
    """
    RankLogger records the rank1 matching accuracy obtained for each
    test dataset at specified evaluation steps and provides a function
    to show the summarized results, which are convenient for analysis.
    Args:
    - source_names (list): list of strings (names) of source datasets.
    - target_names (list): list of strings (names) of target datasets.
    """

    def __init__(self, source_names, target_names):
        self.source_names = source_names
        self.target_names = target_names
        self.logger = {name: {"epoch": [], "Acc": []} for name in self.target_names}

    def write(self, name, epoch, rank1):
        self.logger[name]["epoch"].append(epoch)
        self.logger[name]["Acc"].append(rank1)

    def show_summary(self):
        print("=> Show performance summary")
        for name in self.target_names:
            from_where = "source" if name in self.source_names else "target"
            print(f"{name} ({from_where})")
            for epoch, acc in zip(
                self.logger[name]["epoch"], self.logger[name]["Acc"]
            ):
                print(f"- epoch {epoch}\t Accuracy {acc:.2f}%")
