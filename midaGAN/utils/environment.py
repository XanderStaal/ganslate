# coding=utf-8
# Copyright (c) DIRECT Contributors

import logging
import os
import random
import sys
from os import PathLike
from pathlib import Path
from typing import Optional

import cv2
import numpy as np
import SimpleITK as sitk
import torch
from omegaconf import OmegaConf
from midaGAN.utils import communication, io

logger = logging.getLogger(__name__)


def setup_logging_with_config(conf, debug=False):

    output_dir = Path(conf[conf.mode].output_dir).resolve()
    io.mkdirs(output_dir / conf.mode)

    filename = None
    # Log file only for the global main process
    if communication.get_rank() == 0:
        filename = Path(output_dir) / f"{conf.mode}_log.txt"
    # Stdout for *local* main process only
    use_stdout = communication.get_local_rank() == 0 or debug
    log_level = 'INFO' if not debug else 'DEBUG'

    setup_logging(use_stdout, filename, log_level=log_level)

    logger.info(f'Configuration:\n{OmegaConf.to_yaml(conf)}')
    logger.info(f'Saving checkpoints, logs and config to: {output_dir}')
    logger.info(f'Python version: {sys.version.strip()}')
    logger.info(f'PyTorch version: {torch.__version__}')  # noqa
    logger.info(f'CUDA {torch.version.cuda} - cuDNN {torch.backends.cudnn.version()}')
    logger.info(f'Global rank: {communication.get_rank()}')
    logger.info(f'Local rank: {communication.get_local_rank()}')


def setup_logging(use_stdout: Optional[bool] = True,
                  filename: Optional[PathLike] = None,
                  log_level: Optional[str] = 'INFO') -> None:
    """
    Parameters
    ----------
    use_stdout : bool
        Write output to standard out.
    filename : PathLike
        Filename to write log to.
    log_level : str
        Logging level as in the `python.logging` library.

    Returns
    -------
    None
    """
    if log_level not in ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'EXCEPTION']:
        raise ValueError(f'Unexpected log level got {log_level}.')

    logging.captureWarnings(True)
    log_level = getattr(logging, log_level)

    logger = logging.getLogger()
    logger.setLevel(log_level)

    formatter = logging.Formatter("[%(asctime)s][%(name)s][%(levelname)s] - %(message)s")

    if use_stdout:
        handler = logging.StreamHandler(sys.stdout)
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)

    if filename is not None:
        handler = logging.FileHandler(filename)
        handler.setLevel(log_level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)


def set_seed(seed=0):
    # Inspired also from: https://stackoverflow.com/a/57417097
    logger.info(f"Reproducible mode ON with seed : {seed}")
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)


def setup_threading():
    """
    Sets max threads for SimpleITK and Opencv.
    For numpy etc. set OMP_NUM_THREADS=1 as an env var while running the training script,
    e.g., OMP_NUM_THREADS=1 python tools/train.py ...
    """
    logger.warning("""
    Max threads for SimpleITK and Opencv set to 1
    For numpy etc. set OMP_NUM_THREADS=1 as an env var while running the training script,
    e.g., OMP_NUM_THREADS=1 python tools/train.py ...
    """)
    MAX_THREADS = 1
    sitk.ProcessObject_SetGlobalDefaultNumberOfThreads(MAX_THREADS)
    cv2.setNumThreads(MAX_THREADS)
