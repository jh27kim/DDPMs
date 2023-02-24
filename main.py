import hydra
from logs.create_logger import create_logger
import os

@hydra.main(
    version_base=None,
    config_path="static",  # config file search path is relative to this script
    config_name="config",
)
def main(cfg):
    cmd = 'echo start running > ./logs/main.log'
    os.system(cmd)

    logger = create_logger("DDPM-logger", cfg)
    logger.info(f"Running model : {cfg.network.model}")

if __name__ == "__main__":
    main()
