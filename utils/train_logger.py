# Software Name: attentionless-streaming-asr
# SPDX-FileCopyrightText: Copyright (c) Orange SA
# SPDX-License-Identifier: MIT
#
# This software is distributed under the MIT license,
# see the "LICENSE.txt" file for more details or https://spdx.org/licenses/MIT.html 

import logging

from speechbrain.utils.distributed import if_main_process, main_process_only
from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class TrainLogger:
    """Abstract class defining an interface for training loggers."""

    def log_stats(
        self,
        stats_meta,
        train_stats=None,
        valid_stats=None,
        test_stats=None,
        verbose=False,
    ):
        """Log the stats for one epoch.

        Arguments
        ---------
        stats_meta : dict of str:scalar pairs
            Meta information about the stats (e.g., epoch, learning-rate, etc.).
        train_stats : dict of str:list pairs
            Each loss type is represented with a str : list pair including
            all the values for the training pass.
        valid_stats : dict of str:list pairs
            Each loss type is represented with a str : list pair including
            all the values for the validation pass.
        test_stats : dict of str:list pairs
            Each loss type is represented with a str : list pair including
            all the values for the test pass.
        verbose : bool
            Whether to also put logging information to the standard logger.
        """
        raise NotImplementedError
    
class TensorboardLogger(TrainLogger):
    """Logs training information in the format required by Tensorboard.

    Arguments
    ---------
    save_dir : str
        A directory for storing all the relevant logs.

    Raises
    ------
    ImportError if Tensorboard is not installed.
    """

    def __init__(self, save_dir):
        self.save_dir = save_dir

        # Initialize writer only on main
        self.writer = None
        if if_main_process():
            self.writer = SummaryWriter(self.save_dir)
        self.global_step = {"train": {}, "valid": {}, "meta": 0}

    @main_process_only
    def log_stats(
        self,
        stats_meta,
        train_stats=None,
        valid_stats=None,
        test_stats=None,
        verbose=False,
    ):
        """See TrainLogger.log_stats()"""
        self.global_step["meta"] = stats_meta["epoch"]
        for name, value in stats_meta.items():
            if name != "optimizer" and name != "epoch":
                self.writer.add_scalar(name, value, self.global_step["meta"])

        for dataset, stats in [
            ("train", train_stats),
            ("valid", valid_stats),
        ]:
            if stats is None:
                continue
            for stat, value_list in stats.items():
                if stat not in self.global_step[dataset]:
                    self.global_step[dataset][stat] = 0
                tag = f"{stat}/{dataset}"

                # Both single value (per Epoch) and list (Per batch) logging is supported
                if isinstance(value_list, list):
                    for value in value_list:
                        self.writer.add_scalar(tag, value, self.global_step["meta"])
                else:
                    value = value_list
                    self.writer.add_scalar(tag, value, self.global_step["meta"])

    @main_process_only
    def log_audio(self, name, value, sample_rate):
        """Add audio signal in the logs."""
        self.writer.add_audio(
            name, value, self.global_step["meta"], sample_rate=sample_rate
        )

