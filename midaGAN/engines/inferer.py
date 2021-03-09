import logging

from midaGAN.engines.base import BaseEngineWithInference
from midaGAN.utils import environment
from midaGAN.utils.builders import build_gan, build_loader
from midaGAN.utils.io import decollate
from midaGAN.utils.trackers.inference import InferenceTracker


class Inferer(BaseEngineWithInference):

    def __init__(self, conf):
        super().__init__(conf)
        self.logger = logging.getLogger(type(self).__name__)

        # Logging, dataloader and tracker only when not in deployment mode
        if not self.conf.infer.is_deployment:
            assert self.conf.infer.dataset, "Please specify the dataset for inference."
            environment.setup_logging_with_config(self.conf)
            self.tracker = InferenceTracker(self.conf)
            self.data_loader = build_loader(self.conf)

        self.model = build_gan(self.conf)

    def _set_mode(self):
        self.conf.mode = 'infer'

    def run(self):
        assert not self.conf.infer.is_deployment, \
            "`Inferer.run()` cannot be used in deployment, please use `Inferer.infer()`."

        self.logger.info("Inference started.")

        self.tracker.start_dataloading_timer()
        for i, data in enumerate(self.data_loader, start=1):
            self.tracker.set_iter_idx(i)
            if i == 1:
                input_key = self._get_input_key(data)

                saver = getattr(self.data_loader.dataset, "save", None)
                if saver is None:
                    self.logger.warn(
                        "The dataset class used does not have a 'save' method."
                         " It is not necessary, however, it may be useful in cases"
                         " where the outputs should be stored individually"
                         " ('images/' folder saves input and output in a single image), "
                         " or in a specific format."
                    )

            self.tracker.start_computation_timer()
            self.tracker.end_dataloading_timer()
            out = self.infer(data[input_key])
            self.tracker.end_computation_timer()

            self.tracker.start_saving_timer()
            if saver:
                save_dir = self.output_dir / "saved"
                if "metadata" in data:
                    saver(out, save_dir, metadata=decollate(data["metadata"]))
                else:
                    saver(out, save_dir)
            self.tracker.end_saving_timer()

            visuals = {"input": data[input_key].to(out.device), "output": out}
            len_dataset=len(self.data_loader.dataset)
            self.tracker.log_iter(visuals, len_dataset)

            self.tracker.start_dataloading_timer()

    def _get_input_key(self, data):
        """The dataset (dataloader) needs to return a dict with input data 
        either under the key 'input' or 'A'."""
        if "input" in data:
            return "input"
        elif "A" in data:
            return "A"
        else:
            raise ValueError("An inference dataset needs to provide"
                                "the input data under the dict key 'input' or 'A'.")