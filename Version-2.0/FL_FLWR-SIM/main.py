import hydra
from omegaconf import DictConfig, OmegaConf


@hydra.main(config_path='conf', config_path='base', version_base=None)
def main(cfg: DictConfig):

    # Preparing the dataset
    



if '__name__'=='__main__':
    main()