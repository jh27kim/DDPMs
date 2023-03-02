import sys
import os
import torch

sys.path.append(".")
sys.path.append("..")

from domain.Master import Master
from model.DDPM_network import Unet

class DDPM(Master):
    def __init__(self, cfg, logger):
        super().__init__(cfg, logger)

        self.model = self._create_model()


    def _create_model(self):
        return Unet(dim=64)


    def run(self):
        pass

