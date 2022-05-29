import sys
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import os
import glob

from .dual_encoder import Dual_Encoder

from .dataset import Mewsli_Dataset,Mewsli_Entities_Dataset
from transformers import BertTokenizer
from transformers import logging
logging.set_verbosity_error()

DEVICE = 'cuda:0' if torch.cuda.is_available() else 'cpu'
NUM_WORKERS=2

def compute_embeddings(checkpoint_dir: str, 
                        data_dir: str, 
                        model_type: str, 
                        batch_size: int, 
                        mapping: dict):
    embedd_mentions(checkpoint_dir, data_dir,model_type,batch_size,mapping)
    embedd_entities(checkpoint_dir, data_dir,model_type,batch_size,mapping)


def embedd_mentions(checkpoint_dir: str, data_dir: str, model_type: str, batch_size: int, mapping: dict):
    if mapping is not None:
        dim_size=len(mapping)
    else:
        dim_size=None

    model = Dual_Encoder.load_from_checkpoint(
        checkpoint_path=glob.glob(os.path.join(checkpoint_dir, f'*.ckpt'))[0],
        map_location=torch.device('cpu'),
        model_type=model_type,
        dim_size=dim_size)
    
    model.eval()
    model.to(DEVICE)

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    print('Mentions')
    for split in ['val','test']:
        print(split)
        mentions_dataset = Mewsli_Dataset(data_dir, split=split, tokenizer=tokenizer,model_type=model_type,mapping=mapping)
    
        print(f'Num mentions: {len(mentions_dataset)}')


        mentions_loader = DataLoader(mentions_dataset, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False)
    
        all_embeddings = []
        entity_ids = []
        mention_ids = []
        with torch.no_grad():
            for batch in mentions_loader:
                entity_ids += batch['qids']
                mention_inputs = batch['mention_inputs']
                mention_inputs = mention_inputs.to(DEVICE)
                embeddings = model.get_mention_embeddings(mention_inputs).cpu().numpy()
                all_embeddings.append(embeddings)

        all_embeddings = np.vstack(all_embeddings)
        # print(all_embeddings.shape)
        np.save(f'mewsli_mention_embeddings_{split}', {
            'embeddings': all_embeddings,
            'entity_qids': entity_ids,
            'mention_document_ids': mention_ids})

def embedd_entities(checkpoint_dir: str, data_dir: str, model_type: str, batch_size: int, mapping: dict):
    if mapping is not None:
        dim_size=len(mapping)
    else:
        dim_size=None

    model = Dual_Encoder.load_from_checkpoint(
        checkpoint_path=glob.glob(os.path.join(checkpoint_dir, f'*.ckpt'))[0],
        map_location=torch.device('cpu'),
        model_type=model_type,
        dim_size=dim_size)
    
    model.eval()
    model.to(DEVICE)

    tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    print('\nEntities')
    for split in ['val','test']:
        print(split)
        entities_dataset = Mewsli_Entities_Dataset(data_dir, split=split, tokenizer=tokenizer,model_type=model_type,mapping=mapping)
    
        print(f'Num mentions: {len(entities_dataset)}')


        entities_loader = DataLoader(entities_dataset, batch_size=batch_size, num_workers=NUM_WORKERS, shuffle=False)
    
        all_embeddings = []
        all_ids = []
        with torch.no_grad():
            for batch in entities_loader:
                entities_inputs = batch['entity_inputs'].to(DEVICE)
                entity_embeddings = model.get_entity_embeddings(entities_inputs).cpu().numpy()
                all_embeddings.append(entity_embeddings)
                all_ids += batch['qids']

        all_embeddings = np.vstack(all_embeddings)
        # print(all_embeddings.shape)
        np.save(f'mewsli_entity_embeddings_{split}', {'embeddings': all_embeddings, 'ids': all_ids})
