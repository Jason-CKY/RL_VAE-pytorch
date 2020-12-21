import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from vae import VAE
from dataset.Dataset import ImageDataModule

pl.seed_everything(1234)
checkpoint_callback = ModelCheckpoint(
    monitor='val_elbo',
    dirpath='./checkpoints',
    filename='{epoch:02d}-{elbo-loss:.2f}',
    save_top_k=3,
    mode='min',
)
pl.seed_everything(1234)

datamodule = ImageDataModule(data_dir="dataset/reach_target-vision-v0/wrist_rgb", batch_size=64, val_ratio=0.2)
vae = VAE()
trainer = pl.Trainer(gpus=1, max_epochs=5, progress_bar_refresh_rate=10, callbacks=[checkpoint_callback])
trainer.fit(vae, datamodule)
