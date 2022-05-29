import sys
import os
import numpy as np

def evaluate(embedd_dir: str,split: str):

    m_path = os.path.join(embedd_dir, f'mewsli_mention_embeddings_{split}.npy')
    e_path = os.path.join(embedd_dir, f'mewsli_entity_embeddings_{split}.npy')
    mentions = np.load(m_path, allow_pickle=True)
    entities = np.load(e_path, allow_pickle=True)
    entity_ids = entities.item().get('ids')
    total_entities = len(entity_ids)

    for k in [50,100]:
        retrieval_rate = compute_retrieval_rate(mentions, entities, k=k)
        print(f'Retrieval rate (k={k}):', round(retrieval_rate,5), f', (expected rate for random = {round((k / total_entities),5)})')


def compute_retrieval_rate(mentions, entities, k: int) -> float:
    mention_embeddings = mentions.item().get('embeddings')
    entity_embeddings = entities.item().get('embeddings')
    entity_ids = entities.item().get('ids')
    mention_entity_ids = mentions.item().get('entity_qids')

    scores = np.matmul(mention_embeddings, entity_embeddings.T)

    total_mentions = scores.shape[0]
    n = 0
    for i, row in enumerate(scores):
        indices = np.argsort(row)[::-1][:k]
        true_id = mention_entity_ids[i]
        if true_id in [entity_ids[i] for i in indices]:
            n += 1
    return n / total_mentions