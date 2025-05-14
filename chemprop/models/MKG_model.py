import os
import pandas as pd
import numpy as np
from rdkit import Chem
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
import torch_geometric as pyg

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class MoleculeProcessor:
    FG_FILE_SUFFIX = "_FG.txt"
    MKG_FILE_SUFFIX = "_MKG.txt"
    def __init__(self, input_file, output_dir):
        self.input_file = input_file
        self.output_dir = output_dir
        self.output_fg_file = os.path.join(self.create_output_directory(), f'{self.input_file.split(".")[0]}{self.FG_FILE_SUFFIX}')
        self.output_mkg_file = os.path.join(self.create_output_directory(), f'{self.input_file.split(".")[0]}{self.MKG_FILE_SUFFIX}')

    def read_dataset(self):
        self.dataset = pd.read_csv(self.input_file)
        self.dataset['Index'] = torch.arange(1, len(self.dataset) + 1, device=device)

    def read_external_files(self):
        fg_smarts_file = 'FG_SMARTS.csv'
        external_kg_file = 'FGElement_KG.txt'
        self.fg_smarts_df = pd.read_csv(fg_smarts_file).to(device)
        self.external_kg_df = pd.read_csv(external_kg_file, sep='\t', header=None, names=['head', 'relation', 'tail']).to(device)

    def create_output_directory(self):
        output_data_dir = os.path.join(self.output_dir, 'data', self.input_file.split(".")[0])
        os.makedirs(output_data_dir, exist_ok=True)
        return output_data_dir

    def extract_functional_groups(self, output_fg_file):
        with open(output_fg_file, 'w', encoding='utf-8') as output_fg:
            for index, row in self.dataset.iterrows():
                mol = Chem.MolFromSmiles(row['smiles'])
                if mol:
                    fg_found = False  
                    for fg_index, fg_row in self.fg_smarts_df.iterrows():
                        fg_label, smarts = fg_row['FG_LABEL'], fg_row['SMARTS']
                        if mol.HasSubstructMatch(Chem.MolFromSmarts(smarts)):
                            output_fg.write(f"fg{fg_index + 1}\tisFunctionalGroupOf\tmolecule{row['Index']}\n")
                            fg_found = True 

                    if not fg_found:
                        elements = set([atom.GetSymbol() for atom in mol.GetAtoms()])
                        for element in elements:
                            output_fg.write(f"{element}\tisElementOf\tmolecule{row['Index']}\n")


    def merge_knowledge_graphs(self, output_mkg_file, output_fg_file):
        merged_kg_df = pd.concat([self.external_kg_df, pd.read_csv(self.output_fg_file, sep='\t', header=None, names=['head', 'relation', 'tail'])])
        merged_kg_df.to_csv(self.output_mkg_file, sep='\t', index=False, header=False)


    def process_molecules(self):
        self.read_dataset()
        self.read_external_files()
        self.extract_functional_groups()
        self.merge_knowledge_graphs()

  
class TransDModel(nn.Module):
    def __init__(self, args, file_path):
        super(TransDModel, self).__init__()
        self.args = args
        self.file_path = file_path

        self.hops = self.args.hop
        self.entity_embedding_dim = self.args.entity_embedding_dim
        self.relation_embedding_dim = self.args.relation_embedding_dim
        self.learning_rate = self.args.learning_rate
        self.margin = self.args.margin
        self.criterion = nn.MarginRankingLoss(margin=self.margin).to(device)

        self.load_knowledge_graph()
        self.create_mappings()
        self.all_molecule_nodes = sorted([node for node in self.entity2idx.keys() if node.startswith("molecule")],
                                         key=lambda x: int(x.split('molecule')[1]))
        
        self.projection_matrix = nn.Sequential(
            nn.Linear(self.args.hidden_size, self.args.hidden_size),
            nn.ReLU(),
            nn.Linear(self.args.hidden_size, self.args.entity_embedding_dim)
            ).to(device)

        self.model = self.build_transD_model()
        self.model = self.model.to(device)


    def build_transD_model(self):
        model = nn.ModuleDict({
            'entity_embedding': nn.Embedding(len(self.entity2idx), self.entity_embedding_dim).to(device),
            'relation_embedding': nn.Embedding(len(self.relation2idx), self.relation_embedding_dim).to(device),
            'entity_projection': nn.Embedding(len(self.entity2idx), self.entity_embedding_dim).to(device),
            'relation_projection': nn.Embedding(len(self.relation2idx), self.relation_embedding_dim).to(device),
        }).to(device)

        nn.init.xavier_uniform_(model['entity_embedding'].weight)
        nn.init.xavier_uniform_(model['relation_embedding'].weight)
        nn.init.xavier_uniform_(model['entity_projection'].weight)
        nn.init.xavier_uniform_(model['relation_projection'].weight)

        return model

    def forward(self, train_ind, train_mgraph_embeddings, Data_Length):
        heads, relations, tails = self.triples_to_indices()
        heads, relations, tails = heads.to(device), relations.to(device), tails.to(device)

        e_h = self.model['entity_embedding'](heads)
        e_t = self.model['entity_embedding'](tails)
        r = self.model['relation_embedding'](relations)

        e_h_p = self.model['entity_projection'](heads)
        e_t_p = self.model['entity_projection'](tails)
        r_p = self.model['relation_projection'](relations)

        pos_score = torch.norm(e_h + r - e_t, p=2, dim=1)
        neg_score = torch.norm(e_h_p + r_p - e_t_p, p=2, dim=1)
        target = torch.tensor([-1], dtype=torch.float).to(device)
        loss_kge = self.criterion(pos_score, neg_score, target)

        return loss_kge

    def load_knowledge_graph(self):
        triples = [line.strip().split('\t') for line in open(self.file_path, 'r')]
        self.triples = triples

    def create_mappings(self):
        entities = {triple[0] for triple in self.triples} | {triple[2] for triple in self.triples}
        relations = {triple[1] for triple in self.triples}
        self.entity2idx = {entity: idx for idx, entity in enumerate(entities)}
        self.relation2idx = {relation: idx for idx, relation in enumerate(relations)}

    def triples_to_indices(self):
        heads = [self.entity2idx[triple[0]] for triple in self.triples]
        relations = [self.relation2idx[triple[1]] for triple in self.triples]
        tails = [self.entity2idx[triple[2]] for triple in self.triples]
        return torch.tensor(heads), torch.tensor(relations), torch.tensor(tails)

    def extract_hop_subgraphs(self):
        subgraphs = {}
        G = self.build_graph()  
        for molecule_node in self.all_molecule_nodes:
            subgraph = self.extract_subgraph(G, molecule_node)
            subgraphs[molecule_node] = self.graph_to_pyg_data(subgraph)

        return subgraphs

    def build_graph(self):

        G = nx.Graph()
        for i, triple in enumerate(self.triples):
            head, relation, tail = triple
            if head.startswith("molecule"):
                G.add_node(head, embedding=self.model.entity_embedding(torch.tensor([self.entity2idx[head]], dtype=torch.long).to(device)))
            if tail.startswith("molecule"):
                G.add_node(tail, embedding=self.model.entity_embedding(torch.tensor([self.entity2idx[tail]], dtype=torch.long).to(device)))
            G.add_edge(head, tail, relation=relation)

        return G

    def extract_subgraph(self, G, molecule_node):
        subgraph = nx.ego_graph(G, molecule_node, radius=self.hops)
        non_center_nodes = [node for node in subgraph.nodes if node != molecule_node and node.startswith("molecule")]
        subgraph.remove_nodes_from(non_center_nodes)
        isolated_nodes = [node for node in subgraph.nodes if subgraph.degree[node] == 0]
        subgraph.remove_nodes_from(isolated_nodes)
        return subgraph

    def graph_to_pyg_data(self, graph):
        node_mapping = {node: idx for idx, node in enumerate(graph.nodes)}
        molecule_nodes = [node for node in graph.nodes if node.startswith("molecule")]
        other_nodes = [node for node in graph.nodes if not node.startswith("molecule")]
        ordered_nodes = molecule_nodes + other_nodes

        node_indices = torch.tensor([node_mapping[node] for node in ordered_nodes], dtype=torch.long).to(device)
        x = self.model.entity_embedding(node_indices).detach()

        edge_index = torch.tensor([[node_mapping[edge[0]], node_mapping[edge[1]]] for edge in graph.edges], dtype=torch.long).t().contiguous().to(device)
        data = Data(x=x, edge_index=edge_index)

        return data
