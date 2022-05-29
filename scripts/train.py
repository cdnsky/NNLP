import os
import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import ModelCheckpoint,LearningRateMonitor
from transformers import BertTokenizer
from torch.utils.data import DataLoader
from .dual_encoder import Dual_Encoder
from .dataset import Mewsli_Dataset

from transformers import logging
logging.set_verbosity_error()

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS=2

def train(data_dir: str,
          batch_size: int,
          model_type: str, #'E'/'F'
          mapping = None,
          max_epochs: int = 1):
  
    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    if model_type=='E':
        model = Dual_Encoder(model_type,dim_size=len(mapping))
        trainset = Mewsli_Dataset(data_dir, split='train', tokenizer=tokenizer,model_type=model_type,mapping=mapping)
        valset = Mewsli_Dataset(data_dir, split='val', tokenizer=tokenizer,model_type=model_type,mapping=mapping)

    elif model_type=='F':
        model = Dual_Encoder(model_type)
        trainset = Mewsli_Dataset(data_dir, split='train', tokenizer=tokenizer,model_type=model_type)
        valset = Mewsli_Dataset(data_dir, split='val', tokenizer=tokenizer,model_type=model_type)

    model.to(DEVICE)

    print('Training examples:', len(trainset))
    print('Validation examples:', len(valset))
    valset = [valset[i] for i in range(100)]
    trainloader = DataLoader(trainset, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=True)
    valloader = DataLoader(valset, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False)

    accumulate_grad_batches = max(1, 128 // batch_size)

    logger = pl.loggers.TensorBoardLogger(save_dir='./logs', name='MEL')
    checkpoint_callback = ModelCheckpoint(
            monitor='val_loss',
            mode='min',
            verbose=False,
            dirpath=os.path.join('.', f'checkpoints'),
            filename='{epoch}-{val_loss:.3f}'
        )
    trainer = pl.Trainer(
            gpus=-1 if DEVICE != 'cpu' else 0,
            logger=logger,
            val_check_interval=5,
            accumulate_grad_batches=accumulate_grad_batches,
            log_every_n_steps=5,
            callbacks=[LearningRateMonitor(), checkpoint_callback],
            # callbacks=[checkpoint_callback],
            max_epochs=max_epochs)
    trainer.fit(model, trainloader, valloader)
    return model,trainer