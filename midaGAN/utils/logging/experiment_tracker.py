from pathlib import Path
import logging
import time

import torch
import torchvision
from omegaconf import OmegaConf

from midaGAN.utils import communication, io
from midaGAN.utils.logging.wandb_tracker import WandbTracker
from midaGAN.utils.logging.tensorboard_tracker import TensorboardTracker


class ExperimentTracker:
    def __init__(self, conf):
        self.logger = logging.getLogger(type(self).__name__)
        self.checkpoint_dir = conf.logging.checkpoint_dir

        self.wandb = None
        self.tensorboard = None
        if communication.is_main_process():
            io.mkdirs(Path(self.checkpoint_dir) / 'images')
            self._save_config(conf)
            
            if conf.logging.wandb:
                self.wandb = WandbTracker(conf)

            if conf.logging.tensorboard:
                self.tensorboard = TensorboardTracker(conf)


        self.batch_size = conf.batch_size
        self.log_freq = conf.logging.log_freq
        self.iter_idx = None
        self.iter_end_time = None
        self.iter_start_time = None
        self.t_data = None
        self.t_comp = None

    def _save_config(self, conf):
        config_path = Path(self.checkpoint_dir) / "config.yaml"
        with open(config_path, "w") as file:
            file.write(OmegaConf.to_yaml(conf))

    def set_iter_idx(self, iter_idx):
        self.iter_idx = iter_idx

    def start_computation_timer(self):
        self.iter_start_time = time.time()

    def start_dataloading_timer(self):
        self.iter_end_time = time.time()
    
    def end_computation_timer(self):
        self.t_comp = (time.time() - self.iter_start_time) / self.batch_size
        # reduce computational time data point (avg) and send to the process of rank 0
        self.t_comp = communication.reduce(self.t_comp, average=True, all_reduce=False)

    def end_dataloading_timer(self):
        self.t_data = self.iter_start_time - self.iter_end_time # is it per sample or per batch?
        # reduce data loading per data point (avg) and send to the process of rank 0
        self.t_data = communication.reduce(self.t_data, average=True, all_reduce=False)

    def log_iter(self, learning_rates, losses, visuals):
        """Parameters: # TODO: update this
            iters (int) -- current training iteration
            losses (tuple/list) -- training losses
            t_comp (float) -- computational time per data point (normalized by batch_size)
            t_data (float) -- data loading time per data point (normalized by batch_size)
        """

        if self.iter_idx % self.log_freq == 0:      
            visuals = {k: v for k, v in visuals.items() if v is not None}
            losses = {k: v for k, v in losses.items() if v is not None}
            losses = communication.reduce(losses, average=True, all_reduce=False) # reduce losses (avg) and send to the process of rank 0

            if communication.is_main_process():
                self._log_message(learning_rates, losses) 

                visuals = self._visuals_to_combined_2d_grid(visuals)
                self._save_image(visuals)

                if self.wandb:
                    self.wandb.log_iter(self.iter_idx, learning_rates, losses, visuals)

                if self.tensorboard:
                    self.tensorboard.log_iter(self.iter_idx, learning_rates, losses, visuals)

    def _log_message(self, learning_rates, losses):
        lr_G, lr_D = learning_rates["lr_G"], learning_rates["lr_D"]
        message = '\n' + 20 * '-' + ' '
        message += '(iter: %d | comp: %.3f, data: %.3f | lr_G: %.7f, lr_D = %.7f)' \
                       % (self.iter_idx, self.t_comp, self.t_data, lr_G, lr_D)
        message += ' ' + 20 * '-' +  '\n'
        for k, v in losses.items():
            message += '%s: %.3f ' % (k, v)
        self.logger.info(message)
    
    def _visuals_to_combined_2d_grid(self, visuals):
        # if images are 3D (5D tensors)
        if len(list(visuals.values())[0].shape) == 5: # TODO make nicer
            # Concatenate slices that are at the same level from different visuals along 
            # width (each tensor from visuals.values() is NxCxDxHxW, hence dim=4)
            combined_slices = torch.cat(tuple(visuals.values()), dim=4) 
            combined_slices = combined_slices[0] # we plot a single volume from the batch
            combined_slices = combined_slices.permute(1,0,2,3) # CxDxHxW -> DxCxHxW
            # Concatenate all combined slices along height to form a single 2d image (tensors in tuple are CxHxW, hence dim=1)
            combined_image = torch.cat(tuple(combined_slices), dim=1) 
        else:
            # NxCxHxW
            combined_image = torch.cat(tuple(visuals.values()), dim=3)
            combined_image = combined_image[0]

        combined_image = (combined_image + 1) / 2 # [-1,1] -> [0,1]. Data range important when saving images.
        name = "-".join(visuals.keys()) # e.g. "real_A-fake_B-rec_A-real_B-fake_A-rec_B"
        return {'name': name, 'image': combined_image} # NOTE: image format is CxHxW

    def _save_image(self, visuals):
        name, image = visuals['name'], visuals['image']
        file_path = Path(self.checkpoint_dir) / f"images/{self.iter_idx}_{name}.png" 
        torchvision.utils.save_image(image, file_path)

    def close(self):
        if communication.is_main_process() and self.tensorboard:
            self.tensorboard.close()


