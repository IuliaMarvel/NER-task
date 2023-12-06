import pandas as pd
import hydra
from omegaconf import DictConfig, OmegaConf

from  data_preprocess import get_train_test_loaders_info
from model import get_model
from trainer import get_trainer


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg : DictConfig) -> None:
    ner_data = pd.read_csv(cfg.data_path)
    sentences = list(ner_data.text.apply(lambda s: s.split()))[:1000]
    labels = list(ner_data.labels.apply(lambda s: s.split()))[:1000]
    train_loader, test_loader, vocab_info = get_train_test_loaders_info(sentences, labels)
    model = get_model(
                vocab_info['vocab_size'],
                cfg.model_params.embedding_dim, 
                cfg.model_params.hidden_dim,
                vocab_info['output_size'],
                cfg.training.lr
            )
    trainer = get_trainer(cfg.training.num_epochs)
    trainer.fit(model, train_loader, test_loader)
    print('Evaluating model...')
    trainer.validate(model, test_loader)
    model.plot_training(epochs=cfg.training.num_epochs)


if __name__ == '__main__':
    main()

