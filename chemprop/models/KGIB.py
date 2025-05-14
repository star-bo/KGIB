from argparse import Namespace
import torch
import torch.nn as nn


from .mpn import MPN
from chemprop.nn_utils import get_activation_function, initialize_weights, read_entity2id_file, read_triples_file, add_entity_types

from chemprop.models.SubKG import Subgraph
import random
import os
import numpy as np
import torch_geometric
from torch_geometric.data import Data
from collections import defaultdict, deque
import networkx as nx
import multiprocessing
import matplotlib.pyplot as plt


class MoleculeModel(nn.Module):
    """A MoleculeModel is a model which contains a message passing network following by feed-forward layers."""

    def __init__(self, classification: bool, multiclass: bool, args: Namespace, dim_e):
        """
        Initializes the MoleculeModel.

        :param classification: Whether the model is a classification model.
        """
        super(MoleculeModel, self).__init__()

        self.classification = classification
        if self.classification:
            self.sigmoid = nn.Sigmoid()
        self.multiclass = multiclass
        if self.multiclass:
            self.multiclass_softmax = nn.Softmax(dim=2)
        assert not (self.classification and self.multiclass)

        torch.cuda.set_device(args.gpu)
        self.device = torch.cuda.current_device() 
        self.subgraph_model = Subgraph(args,dim_e).to(self.device)      

        self.visualization_data = [] 
        
        # nn.DataParallel

    def create_encoder(self, args: Namespace):
        """
        Creates the message passing encoder for the model.

        :param args: Arguments.
        """
        self.encoder = MPN(args)

    def create_ffn(self, args: Namespace):
        """
        Creates the feed-forward network for the model.

        :param args: Arguments.
        """
        self.multiclass = args.dataset_type == 'multiclass'
        if self.multiclass:
            self.num_classes = args.multiclass_num_classes
        if args.features_only:
            first_linear_dim = args.features_size
        else:
            # first_linear_dim = args.hidden_size 
            first_linear_dim = args.hidden_size + args.dim_e
            if args.use_input_features:
                first_linear_dim += args.features_dim

        dropout = nn.Dropout(args.dropout)
        activation = get_activation_function(args.activation)

        # Create FFN layers
        if args.ffn_num_layers == 1:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.output_size)
            ]
        else:
            ffn = [
                dropout,
                nn.Linear(first_linear_dim, args.ffn_hidden_size)
            ]
            for _ in range(args.ffn_num_layers - 2):
                ffn.extend([
                    activation,
                    dropout,
                    nn.Linear(args.ffn_hidden_size, args.ffn_hidden_size),
                ])
            ffn.extend([
                activation,
                dropout,
                nn.Linear(args.ffn_hidden_size, args.output_size),
            ])

        # Create FFN model
        self.ffn = nn.Sequential(*ffn)

    def sample_subgraph(self, entity_id, triples_with_scheme, entity_embeddings, num_steps):
        subgraph_nodes = set()  
        subgraph_edges = set()  
        entity_triples = defaultdict(list)

        for head_id, head_type, tail_id, tail_type, rel in triples_with_scheme:
            if tail_type != 'Molecule' or head_id == entity_id:
                entity_triples[head_id].append((tail_id, rel))
            if head_type != 'Molecule' or tail_id == entity_id:
                entity_triples[tail_id].append((head_id, rel))

        current_entity_id = entity_id
        visited_entity_ids = set() 
        visited_entity_ids.add(current_entity_id)

        start_node_index = current_entity_id

        first_hop_neighbors = [(neighbor_id, rel) for neighbor_id, rel in entity_triples[current_entity_id]]
        for neighbor_id, rel in first_hop_neighbors:
            subgraph_edges.add((current_entity_id, neighbor_id, rel))
            subgraph_nodes.add(current_entity_id)
            subgraph_nodes.add(neighbor_id)
            visited_entity_ids.add(neighbor_id)

        nodes_to_explore = list(first_hop_neighbors) 
        while nodes_to_explore and len(subgraph_nodes) < num_steps:
            next_nodes_to_explore = []
            for node, _ in nodes_to_explore:
                neighbors = [(neighbor_id, rel) for neighbor_id, rel in entity_triples[node]
                            if neighbor_id not in visited_entity_ids]

                if neighbors:
                    for neighbor_id, rel in neighbors:
                        if len(subgraph_nodes) >= num_steps:
                            break
                        subgraph_edges.add((node, neighbor_id, rel))
                        subgraph_nodes.add(neighbor_id)
                        visited_entity_ids.add(neighbor_id)
                        next_nodes_to_explore.append((neighbor_id, rel))


                for visited_node in visited_entity_ids:
                    if visited_node != node:  
                        for neighbor_id, rel in entity_triples[visited_node]:
                            if neighbor_id == node:
                                subgraph_edges.add((visited_node, node, rel))
            
            nodes_to_explore = next_nodes_to_explore

        if len(subgraph_nodes) > num_steps:
            sampled_nodes = set(random.sample(subgraph_nodes, num_steps))
            subgraph_edges = {(u, v, r) for u, v, r in subgraph_edges if u in sampled_nodes and v in sampled_nodes}
            subgraph_nodes = sampled_nodes

        subgraph_nodes = list(subgraph_nodes)
        subgraph_edges = list(subgraph_edges)

        node_mapping = {old_index: new_index for new_index, old_index in enumerate(subgraph_nodes)}
        subgraph_edges = [(node_mapping[edge[0]], node_mapping[edge[1]]) for edge in subgraph_edges]

        x = torch.tensor(entity_embeddings[subgraph_nodes], dtype=torch.float).to(self.device)
        edge_index = torch.tensor(subgraph_edges, dtype=torch.long).t().contiguous().to(self.device)

        subgraph_data = Data(x=x, edge_index=edge_index)
        subgraph_data.original_node_indices = subgraph_nodes  
        subgraph_data.start_node_index = start_node_index  

        subgraph_data.triples_with_scheme = triples_with_scheme

        return subgraph_data

    def is_same_molecule_type(self, current_entity_id, neighbor_id, triples_with_scheme):
        """
        Check if the neighbor is of the same molecule type as the current entity.
        """
        current_type = None
        neighbor_type = None

        for head, tail, relation in triples_with_scheme:
            if head == current_entity_id and relation == 'has_type':
                current_type = tail
            if head == neighbor_id and relation == 'has_type':
                neighbor_type = tail
            if current_type and neighbor_type:
                break

        return current_type == neighbor_type

    def process_smiles_list(self, args, input):
        """
        Process the SMILES list, sample subgraphs based on the new strategy, and save as PyG objects.
        """
        mkg_name = args.data_path.split('/')[-1].split('.')[0]
        entity2id_file_path = os.path.join('MKGdata', mkg_name, 'entity2id.txt')
        entity2id = read_entity2id_file(entity2id_file_path)

        triples_file_path = os.path.join('MKGdata', mkg_name, 'train2id.txt')
        triples = read_triples_file(triples_file_path)

        scheme_path = os.path.join('MKGdata', 'scheme.txt')
        triples_with_scheme = add_entity_types(triples, scheme_path) 

        if args.kge is not None:
            if args.kge == 'SimplE':
                entity_embeddings_file_path = f'openke/checkpoint/SimplE300/entity_embeddings_{mkg_name}.npy'
            elif args.kge == 'TransD':
                entity_embeddings_file_path = f'openke/checkpoint/TransD300/entity_embeddings_{mkg_name}.npy'
            elif args.kge == 'TransR':
                entity_embeddings_file_path = f'openke/checkpoint/TransR300/entity_embeddings_{mkg_name}.npy'
            elif args.kge == 'TransH':
                entity_embeddings_file_path = f'openke/checkpoint/TransH300/entity_embeddings_{mkg_name}.npy'
            elif args.kge == 'ComplEx':
                entity_embeddings_file_path = f'openke/checkpoint/ComplEx300/entity_embeddings_{mkg_name}.npy'
            else:
                raise ValueError(f"Unsupported KGE model: {args.kge}")

            entity_embeddings = torch.tensor(np.load(entity_embeddings_file_path), dtype=torch.float).to(self.device)
        else:
            raise ValueError("No KGE model specified. Please provide a valid KGE model.")


        num_steps = args.num_steps

        subgraph_data_list = []
        smiles_list = input[0]
        for smiles in smiles_list:
            entity_id = entity2id.get(smiles)
            if entity_id is None:
                continue 
            subgraph_data = self.sample_subgraph(entity_id, triples_with_scheme, entity_embeddings, num_steps)
            subgraph_data_list.append(subgraph_data)

        return subgraph_data_list



    def contrastive_loss(self, mgraph, mnode, tau):  
        batch_size, _ = mgraph.size() 
        mgraph_abs = mgraph.norm(dim = 1)
        mnode_abs = mnode.norm(dim = 1)

        sim_matrix = torch.einsum('ik,jk->ij', mgraph, mnode) / torch.einsum('i,j->ij', mgraph_abs, mnode_abs)
        sim_matrix = torch.exp(sim_matrix / tau)
        pos_sim = sim_matrix[range(batch_size), range(batch_size)] 
        eps = 1e-8  
        loss = pos_sim / (sim_matrix.sum(dim=1) - pos_sim + eps)
        loss = - torch.log(loss + eps).mean()

        return loss
    
    def forward(self, args, *input):
        """
        Runs the MoleculeModel on input.

        :param input: Input.
        :return: The output of the MoleculeModel.
        """

        mpn_output = self.encoder(*input)
        sample_subKG = self.process_smiles_list(args,input)  
        embeddings, subgraph_output, KL_Loss, preserve_rate, preserved_nodes_list, noisy_nodes_list = self.subgraph_model(sample_subKG) 
        
        cl_loss = self.contrastive_loss(mpn_output,subgraph_output,args.tau)
        concatenated_output = torch.cat((mpn_output, subgraph_output), dim=1)
        output = self.ffn(concatenated_output)      

        if self.classification and not self.training:
            output = self.sigmoid(output)
        if self.multiclass:
            output = output.reshape((output.size(0), -1, self.num_classes)) 
            if not self.training:
                output = self.multiclass_softmax(output)

        if not self.training:
            self.embeddings = concatenated_output.detach().cpu().numpy()
            
        return output, KL_Loss, cl_loss, preserve_rate, preserved_nodes_list, noisy_nodes_list 

def build_model(args: Namespace) -> nn.Module:
    """
    Builds a MoleculeModel, which is a message passing neural network + feed-forward layers.

    :param args: Arguments.
    :return: A MoleculeModel containing the MPN encoder along with final linear layers with parameters initialized.
    """
    output_size = args.num_tasks
    args.output_size = output_size
    if args.dataset_type == 'multiclass':
        args.output_size *= args.multiclass_num_classes

    model = MoleculeModel(classification=args.dataset_type == 'classification', multiclass=args.dataset_type == 'multiclass', args= args, dim_e=args.dim_e)
    model.create_encoder(args)
    model.create_ffn(args)

    initialize_weights(model)

    return model