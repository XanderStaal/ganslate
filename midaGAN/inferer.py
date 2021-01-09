import logging

from monai.inferers import SlidingWindowInferer

from midaGAN.data import build_loader
from midaGAN.data.utils import decollate
from midaGAN.nn.gans import build_gan
from midaGAN.utils.trackers.inference_tracker import InferenceTracker


class Inferer():

    def __init__(self, conf):
        self.logger = logging.getLogger(type(self).__name__)
        self.conf = conf
        self.data_loader = build_loader(self.conf)
        self.tracker = InferenceTracker(self.conf)
        self.model = build_gan(self.conf)
        self.sliding_window_inferer = self._init_sliding_window_inferer()

    def run(self):
        has_metadata = False
        inference_dir = self.conf.logging.inference_dir
        self.logger.info("Inference started.")

        self.tracker.start_dataloading_timer()
        for i, data in enumerate(self.data_loader, start=1):
            # Sometimes, metadata is necessary to be able to store the generated outputs.
            # E.g. origin, spacing and direction is required in order to properly save medical images.
            # Dataloader yields a list when passing multiple values at once.
            if isinstance(data, list):
                has_metadata = True
                data, metadata = data
                metadata = decollate(metadata)

            self.tracker.start_computation_timer()
            self.tracker.end_dataloading_timer()
            out = self.infer(data)
            self.tracker.end_computation_timer()

            self.tracker.start_saving_timer()
            # Inference-time dataset class has to have a `save()` method
            if has_metadata:
                self.data_loader.dataset.save(out, metadata, inference_dir)
            else:
                self.data_loader.dataset.save(out, inference_dir)
            self.tracker.end_saving_timer()

            self.tracker.log_message(i, self.data_loader)
            self.tracker.start_dataloading_timer()

    def infer(self, data):
        data = data.to(self.model.device)

        # Sliding window (i.e. patch-wise) inference
        if self.sliding_window_inferer:
            return self.sliding_window_inferer(data, self.model.infer)
        return self.model.infer(data)

    def _init_sliding_window_inferer(self):
        if not self.conf.sliding_window:
            return
        return SlidingWindowInferer(roi_size=self.conf.sliding_window.window_size,
                                    sw_batch_size=self.conf.sliding_window.batch_size,
                                    overlap=self.conf.sliding_window.overlap,
                                    mode=self.conf.sliding_window.mode,
                                    cval=-1)
