import hydra
from logs.create_logger import create_logger
import os
from domain.Master import Master
from domain.DDPM import DDPM

@hydra.main(
    version_base=None,
    config_path="static",  # config file search path is relative to this script
    config_name="config",
)
def main(cfg):
    cmd = 'echo Start running > ./logs/main.log'
    os.system(cmd)

    cur_dir = os.getcwd()
    logger = create_logger("DDPM-logger", cfg)
    model = None    

    if cfg.network.model.lower() == "ddpm":
        logger.info(f"Training model : {cfg.network.model}")
        model = DDPM(cfg, logger, cur_dir)
    
    else:
        raise NotImplementedError()

    if not cfg.test:
        model.train()
    else:
        model.sample()


if __name__ == "__main__":
    main()
