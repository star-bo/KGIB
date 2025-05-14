import torch
import torch.nn.functional as F
from argparse import Namespace


import torch
import torch.nn.functional as F
from argparse import Namespace
import networkx as nx
import matplotlib.pyplot as plt

torch.autograd.set_detect_anomaly(True)


class SAGE(torch.nn.Module):
    """
    SAGE layer class.
    """
    def __init__(self, args: Namespace, number_of_features):
        """
        Creating a SAGE layer.
        :param args: Arguments object.
        :param number_of_features: Number of node features.
        """
        super(SAGE, self).__init__()
        self.args = args
        self.number_of_features = number_of_features
        self.device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
        self.gumbeltau = args.gumbeltau 
        self._setup()


    def _setup(self):
        """
        Setting up upstream and pooling layers.
        """
        self.fully_connected_1 = torch.nn.Linear(self.number_of_features,
                                                 self.args.first_dense_neurons).to(self.device)

        self.fully_connected_2 = torch.nn.Linear(self.args.first_dense_neurons,
                                                 self.args.second_dense_neurons).to(self.device)
        
        self.fully_connected_3 = torch.nn.Linear(self.args.second_dense_neurons,
                                                 self.args.third_dense_neurons).to(self.device)


    def forward(self, data):
        """
        Making a forward pass with the graph level data.
        :param data: Data feed dictionary.
        :return graph_embedding: Graph level embedding.
        :return penalty: Regularization loss.
        """
        features = data.x.to(self.device) 
        epsilon = 1e-7
        num_nodes = features.size()[0]

        node_feature = features
        static_node_feature = node_feature.clone().detach()
        node_feature_std, node_feature_mean = torch.std_mean(static_node_feature, dim=0)

        abstract_features_1 = torch.tanh(self.fully_connected_1(node_feature))  
        abstract_features_2 = torch.tanh(self.fully_connected_2(abstract_features_1))
        assignment = torch.nn.functional.softmax(self.fully_connected_3(abstract_features_2), dim=1)

        gumbel_assignment = self.gumbel_softmax(assignment)  
        graph_feature = torch.sum(node_feature, dim = 0, keepdim=True)
        node_feature_mean = node_feature_mean.repeat(num_nodes,1)

        lambda_pos = gumbel_assignment[:,0].unsqueeze(dim = 1)
        lambda_neg = gumbel_assignment[:,1].unsqueeze(dim = 1)

        subgraph_representation = torch.sum(lambda_pos * node_feature, dim = 0, keepdim=True) 

        noisy_node_feature_mean = lambda_pos * node_feature + lambda_neg * node_feature_mean
        noisy_node_feature_std = lambda_neg * node_feature_std

        noisy_node_feature = noisy_node_feature_mean + torch.rand_like(noisy_node_feature_mean) * noisy_node_feature_std
        noisy_graph_feature = torch.sum(noisy_node_feature, dim = 0, keepdim=True)

        KL_tensor = 0.5 * ((noisy_node_feature_std ** 2) / (node_feature_std + epsilon) ** 2) + \
                    torch.sum(((noisy_node_feature_mean -node_feature_mean)/(node_feature_std + epsilon))**2, dim = 0) 
        KL_Loss = torch.mean(KL_tensor)

        preserve_rate = torch.sum(assignment[:,0] > 0.5) / assignment.size()[0]
        
        preserved_nodes = assignment[:, 0] > 0.5
        noisy_nodes = ~preserved_nodes

        return graph_feature, noisy_graph_feature, subgraph_representation, KL_Loss, preserve_rate, preserved_nodes, noisy_nodes
    
    def gumbel_softmax(self, prob):
        return F.gumbel_softmax(prob, tau=self.gumbeltau, dim = -1)  

class Subgraph(torch.nn.Module): 
    def __init__(self, args, dim_e):
        super(Subgraph, self).__init__()

        self.args = args
        self.dim_e = dim_e
        self.device = torch.cuda.set_device(args.gpu)
        self._setup()

    def _setup(self):
        self.graph_level_model = SAGE(self.args, self.dim_e).to(self.device)

    def forward(self, graphs):
        embeddings = [] 
        positive = [] 
        subgraph = [] 

        preserve_rate = 0
        KL_Loss = 0
        preserved_nodes_list = []
        noisy_nodes_list = []

        for graph in graphs:
            embedding, noisy, subgraph_emb, kl_loss, one_preserve_rate, preserved_nodes, noisy_nodes = self.graph_level_model(graph)
            embeddings.append(embedding)
            positive.append(noisy)
            subgraph.append(subgraph_emb)
            
            KL_Loss += kl_loss
            preserve_rate += one_preserve_rate
            preserved_nodes_list.append(preserved_nodes)
            noisy_nodes_list.append(noisy_nodes)

        embeddings = torch.cat(tuple(embeddings),dim = 0)
        positive = torch.cat(tuple(positive),dim = 0)
        subgraph = torch.cat(tuple(subgraph),dim = 0)
        
        KL_Loss = KL_Loss / len(graphs)
        preserve_rate = preserve_rate / len(graphs)

        del embedding, noisy, subgraph_emb
        torch.cuda.empty_cache()

        return embeddings, positive, KL_Loss, preserve_rate, preserved_nodes_list, noisy_nodes_list

