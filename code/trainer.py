import pytorch_lightning as pl

def get_trainer(max_epochs):
    trainer = pl.Trainer(max_epochs=max_epochs, enable_progress_bar=True)
    return trainer
