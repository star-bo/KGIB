import logging
from argparse import Namespace
from typing import Callable, List, Union

from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler
from tqdm import trange
import numpy as np

from chemprop.data import MoleculeDataset
from chemprop.nn_utils import compute_gnorm, compute_pnorm, NoamLR

from openke.config import Trainer
from openke.module.model import TransD, TransE, TransH, TransR, SimplE, ComplEx, RESCAL, RotatE, Analogy, DistMult, HolE, DistMult
from openke.module.loss import MarginLoss, SigmoidLoss
from openke.module.strategy import NegativeSampling
from openke.data import TrainDataLoader
from chemprop.train.kge_train import train_transh, train_transr, train_transd, train_simple, train_rotate, train_distmult_adv, train_complex, train_analogy, train_rescal


def train(model: nn.Module,
          data: Union[MoleculeDataset, List[MoleculeDataset]],
          loss_func: Callable,
          optimizer: Optimizer,
          scheduler: _LRScheduler,
          args: Namespace,
          n_iter: int = 0,
          logger: logging.Logger = None,
          writer: SummaryWriter = None) -> int:
    """
    Trains a model for an epoch.

    :param model: Model.
    :param data: A MoleculeDataset (or a list of MoleculeDatasets if using moe).
    :param loss_func: Loss function.
    :param optimizer: An Optimizer.
    :param scheduler: A learning rate scheduler.
    :param args: Arguments.
    :param n_iter: The number of iterations (training examples) trained on so far.
    :param logger: A logger for printing intermediate results.
    :param writer: A tensorboardX SummaryWriter.
    :return: The total number of iterations (training examples) trained on so far.
    """
    debug = logger.debug if logger is not None else print

    mkg_name = args.data_path.split('/')[-1].split('.')[0]
    if args.kge_training and n_iter == 0:

        kge_model = train_transd(
            args=args,
            dim_e=300,
            dim_r=300,
            margin=4.0,
            nbatches=100,
            train_times=1000,
            alpha=1.0,
            use_gpu=True,
            ckpt_dir="openke/checkpoint/TransD300",
        )
        
        # 保存实体和关系的嵌入向量
        entity_embeddings = kge_model.ent_embeddings.weight.data.cpu().numpy()
        relation_embeddings = kge_model.rel_embeddings.weight.data.cpu().numpy()  
        entity_embeddings_path = f'openke/checkpoint/TransD300/entity_embeddings_{mkg_name}.npy'
        relation_embeddings_path = f'openke/checkpoint/TransD300/relation_embeddings_{mkg_name}.npy'
        np.save(entity_embeddings_path, entity_embeddings)
        np.save(relation_embeddings_path, relation_embeddings)

        entity_embeddings = np.load(entity_embeddings_path)
        relation_embeddings = np.load(relation_embeddings_path)
    
    model.train()
    
    data.shuffle()

    loss_sum, iter_count = 0, 0

    num_iters = len(data) // args.batch_size * args.batch_size  # don't use the last batch if it's small, for stability

    iter_size = args.batch_size

    for i in trange(0, num_iters, iter_size):
        # Prepare batch
        if i + args.batch_size > len(data):
            break
        mol_batch = MoleculeDataset(data[i:i + args.batch_size])
        smiles_batch, features_batch, target_batch = mol_batch.smiles(), mol_batch.features(), mol_batch.targets()
        batch = smiles_batch
        mask = torch.Tensor([[x is not None for x in tb] for tb in target_batch])
        targets = torch.Tensor([[0 if x is None else x for x in tb] for tb in target_batch])

        if next(model.parameters()).is_cuda:
            mask, targets = mask.cuda(), targets.cuda()

        class_weights = torch.ones(targets.shape)

        if args.cuda:
            class_weights = class_weights.cuda()

        # Run model
        model.zero_grad()
        preds, MI_Loss, cl_loss, preserve_rate, preserved_nodes_list, noisy_nodes_list = model(args, batch, features_batch)

        if args.dataset_type == 'multiclass':
            targets = targets.long()
            loss = torch.cat([loss_func(preds[:, target_index, :], targets[:, target_index]).unsqueeze(1) for target_index in range(preds.size(1))], dim=1) * class_weights * mask
        else:
            loss = loss_func(preds, targets) * class_weights * mask 

        loss = loss.sum() / mask.sum()
        loss += args.mi_weight * MI_Loss + args.cl_weight * cl_loss
        
        loss_sum += loss.item()

        iter_count += len(mol_batch)
        
        loss.backward()
        optimizer.step()


        if isinstance(scheduler, NoamLR):
            scheduler.step()

        n_iter += len(mol_batch)

        # Log and/or add to tensorboard
        if (n_iter // args.batch_size) % args.log_frequency == 0:
            lrs = scheduler.get_lr()
            pnorm = compute_pnorm(model)
            gnorm = compute_gnorm(model)
            loss_avg = loss_sum / iter_count
            loss_sum, iter_count = 0, 0

            if writer is not None:
                writer.add_scalar('train_loss', loss_avg, n_iter)
                writer.add_scalar('param_norm', pnorm, n_iter)
                writer.add_scalar('gradient_norm', gnorm, n_iter)
                for i, lr in enumerate(lrs):
                    writer.add_scalar(f'learning_rate_{i}', lr, n_iter)

    return n_iter
