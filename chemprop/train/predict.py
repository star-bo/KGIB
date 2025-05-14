from typing import List
import torch
import torch.nn as nn
from tqdm import trange
from chemprop.data import MoleculeDataset, StandardScaler
from argparse import Namespace
import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from chemprop.data.scaffold import generate_scaffold
from sklearn.metrics import davies_bouldin_score
from collections import defaultdict
import json
from sklearn.preprocessing import StandardScaler

from tqdm import trange


def predict(model: nn.Module,
            data: MoleculeDataset,
            batch_size: int,
            args: Namespace,
            scaler: StandardScaler = None) -> List[List[float]]:
    """
    Makes predictions on a dataset using an ensemble of models.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param scaler: A StandardScaler object fit on the training targets.
    :return: A list of lists of predictions. The outer list is examples
    while the inner list is tasks.
    """
    model.eval()
    preds = []

    num_iters, iter_step = len(data), batch_size

    for i in range(0, num_iters, iter_step):
        # Prepare batch
        mol_batch = MoleculeDataset(data[i:i + batch_size])
        smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()

        # Run model
        batch = smiles_batch

        with torch.no_grad():
            batch_preds, KL_Loss, cl_loss, preserve_rate, preserved_nodes_list, noisy_nodes_list = model(args, batch, features_batch)

        batch_preds = batch_preds.data.cpu().numpy()
        
        # Inverse scale if regression
        if scaler is not None:
            batch_preds = scaler.inverse_transform(batch_preds)

        # Collect vectors
        batch_preds = batch_preds.tolist()
        preds.extend(batch_preds)

    return preds

def get_embeddings(model: nn.Module,
                   data: MoleculeDataset,
                   batch_size: int,
                   args: Namespace) -> List[List[float]]:
    """
    Retrieves embeddings for a dataset using a model.

    :param model: A model.
    :param data: A MoleculeDataset.
    :param batch_size: Batch size.
    :param args: Namespace object containing additional arguments.
    :return: A list of lists of embeddings.
    """
    model.eval()
    all_embeddings = []

    num_iters, iter_step = len(data), batch_size

    for i in range(0, num_iters, iter_step):
        mol_batch = MoleculeDataset(data[i:i + batch_size])
        smiles_batch, features_batch = mol_batch.smiles(), mol_batch.features()

        with torch.no_grad():
            output, KL_Loss, cl_loss, preserve_rate, preserved_nodes_list, noisy_nodes_list = model(args, smiles_batch, features_batch)
        
        embeddings = model.embeddings
        all_embeddings.extend(embeddings.tolist())
   
    return all_embeddings


def create_combined_dataset(train_data, val_data, test_data):
    combined_smiles = train_data.smiles() + val_data.smiles() + test_data.smiles()
    combined_targets = train_data.targets() + val_data.targets() + test_data.targets()
    return MoleculeDataset(smiles=combined_smiles, targets=combined_targets)
