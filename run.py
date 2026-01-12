import hydra
from omegaconf import DictConfig
from src.classifier import train, evaluate
from src.feature_extraction import extract_features

@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    if "transcribe" in cfg.steps:
        extract_features.run(cfg.steps.feature_extraction, transcribe_mode=True)
        evaluate.run(cfg.steps.evaluate, transcribe_mode=True)
        print('Transcription complete.')

    elif "train" in cfg.steps:
        extract_features.run(cfg.steps.feature_extraction)
        train.run(cfg.steps.train)
        print('Training complete.')

    elif "evaluate" in cfg.steps:
        extract_features.run(cfg.steps.feature_extraction)
        evaluate.run(cfg.steps.evaluate)
        print('Evaluation complete.')


if __name__ == "__main__":
    main()

