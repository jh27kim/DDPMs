from dataset.DatasetClass import DatasetClass as Dataset
from torch.utils.data import DataLoader
import torch


class Master():
    def __init__(self, cfg, logger, cur_dir):
        self.cfg= cfg
        self.logger = logger
        self.cur_dir = cur_dir

        self.data_folder = cfg.data.datapath
        self.image_size = cfg.data.imgsize
        self.batch_size = cfg.data.batchsize

        self._init_cuda()
        self._load_dataset()


    def _init_cuda(self):
        if torch.cuda.is_available():
            device_id = self.cfg.cuda

            if device_id > torch.cuda.device_count() - 1:
                self.logger.warn("Invalid device ID. " f"There are {torch.cuda.device_count()} devices but got index {device_id}.")

                device_id = 0
                self.cfg.cuda = device_id

                self.logger.info(f"Set device ID to {self.cfg.cuda.device_id} by default.")
                
            torch.cuda.set_device(self.cfg.cuda)
            self.device = torch.cuda.current_device()
            self.logger.info(f"CUDA device detected. Using device {torch.cuda.current_device()}.")

        else:
            self.logger.warn("CUDA is not supported on this system. Using CPU by default.")


    def _load_dataset(self):
        self.dataset = Dataset(self.cur_dir, self.data_folder, self.image_size, augment_horizontal_flip=True)
        assert self.dataset[0].shape[1] == self.dataset[0].shape[2]
        self.dataloader = DataLoader(self.dataset, batch_size = self.batch_size, shuffle = True, pin_memory = True)
