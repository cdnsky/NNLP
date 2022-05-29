import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torch import nn, Tensor
from transformers import BertModel, AdamW
from typing import Dict

class Dual_Encoder(pl.LightningModule):
    def __init__(self, model_type,dim_size=None): #dim_size - len(qid_to_ind)
        super(Dual_Encoder, self).__init__()

        self.model_type = model_type

        # Mention embedder
        pretrained_model_m = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.mention_embedding_part = pretrained_model_m.embeddings
        self.mention_main_part_4_layers = pretrained_model_m.encoder.layer[:4]
        # Entity embedder
        pretrained_model_e = BertModel.from_pretrained('bert-base-multilingual-cased')
        self.entity_main_part_4_layers = pretrained_model_e.encoder.layer[:4]
        if self.model_type=='E':
            self.entity_embedding_part = nn.Embedding(dim_size, 768)
        elif self.model_type=='F':
            self.entity_embedding_part = pretrained_model_e.embeddings

        self.fc_me = nn.Linear(768, 128)
        self.fc_ee = nn.Linear(768, 128)

    def get_entity_embeddings(self, entity_inputs):
        entity_inputs = entity_inputs.to(self.device)

        start_embeds = self.entity_embedding_part(entity_inputs).squeeze(1)
        tmp = self.entity_main_part_4_layers[0](start_embeds)
        for i in range(1,4):
            tmp = self.entity_main_part_4_layers[i](tmp[0])
        ee = tmp[0][:, 0]
        ee = self.fc_ee(ee)
        return ee

    def get_mention_embeddings(self, mention_inputs):
        mention_inputs = mention_inputs.to(self.device)
        
        start_embeds = self.mention_embedding_part(mention_inputs)
        tmp = self.mention_main_part_4_layers[0](start_embeds)
        for i in range(1,4):
            tmp = self.mention_main_part_4_layers[i](tmp[0])    
        me = tmp[0][:, 0]
        me = self.fc_me(me)
        return me

    def forward(self, mention_inputs, entity_inputs=None, **kwargs):
        me = self.get_mention_embeddings(mention_inputs)

        if entity_inputs is not None:
            ee = self.get_entity_embeddings(entity_inputs)
            return me, ee

        return me

    def training_step(self, batch, batch_idx):
        me, ee = self.forward(**batch)
        scores = me.mm(ee.t())

        bs = ee.size(0)
        target = torch.LongTensor(torch.arange(bs))
        target = target.to(self.device)
        loss = F.cross_entropy(scores, target, reduction="mean")
        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        me, ee = self.forward(**batch)
        scores = me.mm(ee.t())

        bs = ee.size(0)
        target = torch.LongTensor(torch.arange(bs))
        target = target.to(self.device)
        loss = F.cross_entropy(scores, target, reduction="mean")
        return {'val_loss': loss}

    def validation_epoch_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        self.log('val_loss', avg_loss)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=2e-5)
        return optimizer