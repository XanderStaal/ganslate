# coding=utf-8
# Copyright (c) midaGAN Contributors
import random 
import numpy as np

from midaGAN.utils import sitk_utils
import logging

logger = logging.getLogger(__name__)

def size_invalid_check_and_replace(volume, patch_size, replacement_paths=[], original_path=None):
    """
    Check if volume loaded is invalid, if so replace with another volume from the same patient. 

    Parameters
    ----------------
    volume: Input volume to check for validity
    patch_size: Patch size to compare against. If volume smaller than patch size, it is considered invalid
    replacement_paths: List of paths to sample from if volume is invalid
    original_path: Path of current volume. Used to remove entry


    Returns
    ----------------
    volume or None

    """

    if original_path is not None:
        replacement_paths.remove(original_path)

    # Check if volume is smaller than patch size
    
    if len(patch_size) == 3:
        fn = eval(f"sitk_utils.is_volume_smaller_than")
    elif len(patch_size) == 2:
        fn = eval(f"sitk_utils.is_image_smaller_than")
    else:
        raise NotImplementedError()

    while fn(volume, patch_size):
        logger.warning(f"Volume size smaller than the defined patch size.\
            Volume: {sitk_utils.get_size_zxy(volume)} \npatch_size: {patch_size}. \n \
            Volume path: {original_path}")

        logger.warning(f"Replacing with random choice from: {replacement_paths}")
        
        if len(replacement_paths) == 0:
            return None

        # Load volume randomly from replacement paths
        path = random.choice(replacement_paths)
        logger.warning(f"Loading replacement scan from {path}")
        volume = sitk_utils.load(path)  

        # Remove current path from replacement paths
        replacement_paths.remove(path)

    return volume


def pad(index, volume):
    assert len(index) == len(volume.shape)
    pad_width = [(0,0) for _ in range(len(index))]  # by default no padding

    for dim in range(len(index)):
        if index[dim] > volume.shape[dim]:
            pad = index[dim] - volume.shape[dim]
            pad_per_side = pad // 2
            pad_width[dim] = (pad_per_side, pad % 2 + pad_per_side)  

    return np.pad(volume, pad_width, 'constant', constant_values=volume.min())