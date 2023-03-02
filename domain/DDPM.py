import sys
import os
import torch
from tqdm import tqdm
from torch import nn
import torch.nn.functional as F

sys.path.append(".")
sys.path.append("..")

from domain.Master import Master
from model.DDPM_network import Unet
import utils.DDPM_utils as DDPM_utils

class DDPM(Master):
    def __init__(self, cfg, logger, cur_dir):
        super().__init__(cfg, logger, cur_dir)

        self.model = self._create_model().to(self.device)

        self._initialize_constant()


    def _create_model(self):
        return Unet(dim=64)

    
    def _initialize_constant(self):
        # Forward Process constants
        self.beta = torch.linspace(self.cfg.network.train.beta1, self.cfg.network.train.beta2, self.cfg.network.train.timestep, device=self.device)
        self.alpha = 1. - self.beta
        self.alpha_bar = torch.cumprod(self.alpha, dim=0)

        self.sqrt_alpha_bar = torch.sqrt(self.alpha_bar)
        self.sqrt_one_minus_alpha_bar =torch.sqrt(1. - self.alpha_bar)

        # Train parameters
        self.lr = self.cfg.network.train.lr
        self.steps = self.cfg.network.train.steps
        self.timestep = self.cfg.network.train.timestep

        # Output path
        self.ckpt_dir = os.path.join(self.cur_dir, self.cfg.logs.ckpt_dir)
        if not os.path.exists(self.ckpt_dir):
            os.makedirs(self.ckpt_dir)

        self.output_dir = os.path.join(self.cur_dir, self.cfg.output_dir)
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)

        self.ckpt_epoch = self.cfg.logs.ckpt_epoch
        self.ckpt_file = self.cfg.logs.ckpt_file
        self.viz_epoch = self.cfg.logs.visualize_epoch

    
    def _initialize_loss(self):
        self.loss_fn = None

        if self.cfg.network.train.loss.lower() == "mse":
            self.loss_fn = torch.nn.MSELoss()

        else:
            raise NotImplementedError()

    def _initialize_optimizer(self):
        self.optimizer = None

        if self.cfg.network.train.optimizer.lower() == "adam":
            self.optimizer = torch.optim.Adam(self.model.parameters(), self.lr, betas=(0.9, 0.99))
        
        else:
            raise NotImplementedError()


    def train(self):
        self._initialize_loss()
        self._initialize_optimizer()
        start = self._load_ckpt()

        for epoch in tqdm(range(start, self.steps // len(self.dataset))):
            total_loss = 0.
            for img in self.dataloader:
                self.optimizer.zero_grad()

                batch, channel, height, width = img.shape

                t = torch.randint(0, self.timestep, (batch,), device=self.device)
                eps = torch.randn_like(img, device=self.device)
                img = DDPM_utils.normalize_to_neg_one_to_one(img).to(self.device)

                first_coeff = torch.gather(self.sqrt_alpha_bar, 0, t).reshape(batch, *((1,) * (len(img.shape) - 1)))
                second_coeff = torch.gather(self.sqrt_one_minus_alpha_bar, 0, t).reshape(batch, *((1,) * (len(img.shape) - 1)))

                xt = first_coeff * img + second_coeff * eps
                eps_pred = self.model(xt, t)

                loss = self.loss_fn(eps_pred, eps)
                total_loss += loss.item()
                loss.backward()

                self.optimizer.step()


            self.logger.info(f"Epoch: {epoch}.  Total loss : {total_loss}")

            if (epoch + 1) % self.ckpt_epoch:
                self._save_ckpt(epoch)

            
            if (epoch + 1) % self.viz_epoch:
                self._visualize(epoch)


    def sample(self, batch=4):
        epoch = self._load_ckpt()
        self.logger.info(f"Starting sampling using epoch at {epoch}")
        
        img_size = self.dataset.get_img_height

        imgs = []
        xt = torch.randn((batch, 3, img_size, img_size), device=self.device)
        for t in tqdm(reversed(range(0, self.timestep))):
            xt, x0 = self._p_sample(xt, t, batch, img_size)
            imgs.append(xt)



    def _p_sample(self, xt, t, batch, img_size):
        
    

    def _save_ckpt(self, epoch):
            ckpt_file = os.path.join(self.ckpt_dir, f"ddpm_ckpt_{str(epoch).zfill(6)}.pth")

            ckpt = {"epoch": epoch}
            ckpt["model"] = self.model_coarse.state_dict()
            if self.optimizer is not None:
                ckpt["optimizer_state_dict"] = self.optimizer.state_dict()

            torch.save(ckpt, ckpt_file)


    def _load_ckpt(self):
        epoch = 0
        ckpt_file = os.path.join(self.ckpt_dir, self.ckpt_file)

        if not os.path.exists(ckpt_file):
            self.logger.warn("Checkpoint not found, proceeding at epoch 0")
            return epoch
        
        self.logger.info(f"Checkpoint found. Loading {ckpt_file}")

        ckpt = torch.load(ckpt_file, map_location="cpu")
        
        epoch = ckpt["epoch"]
        self.model.load_state_dict(ckpt["model"]).to(torch.cuda.current_device())
        if self.optimizer is not None:
            self.optimizer.load_state_dict(ckpt["optimizer_state_dict"])

        self.logger.info("Check point loaded. Epoch at : ", epoch)

        return epoch


    
    def _visualize(self, epoch):
        save_dir = os.path.join(self.output_dir, str(epoch).zfill(5))
        if not os.path.exists(save_dir):
            os.makedirs(save_dir, exist_ok=True)

        with torch.no_grad():
            sampled_img = self.sample(4)
            torchvision.utils.save_image(sampled_img, os.path.join(save_dir, f"img_{i}.png"))
