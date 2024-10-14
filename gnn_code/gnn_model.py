import torch
import torch.nn as nn
import torch_geometric.nn as geom_nn

import ipdb


class HeterogeneousGraphClassifier(nn.Module):
    '''
    if model_type = 'sage': Heterogeneous GraphSAGE graph classifier
    Creates the homogeneous GraphSAGE with SAGEConv, converts it to hetero, then uses the hetero form

    if model_type = 'hgt': Heterogeneous graph transformer graph classifier
    Uses HGTConv layers to process a heterogeneous graph

    Creates node embeddings that are then used for graph representation
    '''
    def __init__(self, input_dim, hidden_dim, num_classes, 
                 num_conv_layers, num_lin_layers, hetero_metadata, graph_metadata, 
                 class_node, lin_layer_feats=-1, dropout=0.0, include_basis=True, include_covs=True, do_global_pool='none',
                 model_type='hgt'):
        super().__init__()

        # Variables to store values
        self.in_dim = input_dim
        self.h_dim = hidden_dim
        self.out_dim = num_classes
        self.num_conv_layers = num_conv_layers
        self.num_lin_layers = num_lin_layers
        self.class_node = class_node
        self.hetero_metadata = hetero_metadata
        self.graph_metadata = graph_metadata
        
        self.lin_layer_feats = lin_layer_feats
        self.include_basis = include_basis
        self.include_covs = include_covs
        self.do_global_pool = do_global_pool
        self.model_type = model_type
        self.drop_prob = dropout

        # Variable to shift output when performing explanations
        self.explaining = False


        if(self.include_basis and not self.model_type == 'sage'):
            self.basis_lins = nn.ModuleDict()
            for ntype in self.graph_metadata.keys():
                self.basis_lins[ntype] = nn.LazyLinear(self.in_dim)
        

        if(self.model_type == 'sage'):
            # Get converted homogeneous GraphSAGE model for convolutions
            self.hetero_gnn = geom_nn.to_hetero_with_bases(
                Homogeneous_GraphSAGE_Convs(self.in_dim, 
                    self.h_dim, 
                    self.num_conv_layers,
                    self.drop_prob),
                metadata=hetero_metadata,
                num_bases=3,
                in_channels={'x_dict': self.in_dim}) #note in_channels is for a specific input being passed in to forward (like x_dict)
        
        elif(self.model_type == 'hgt'):
            # Get HGT convolutions model for convolutions
            self.hetero_gnn = HGT_Convs(self.in_dim, 
                self.h_dim, 
                self.num_conv_layers,
                self.drop_prob,
                self.hetero_metadata)
            
        elif(self.model_type == 'han'):
            # Get HAN convolutions model for convolutions
            self.hetero_gnn = HAN_Convs(self.in_dim, 
                self.h_dim, 
                self.num_conv_layers,
                self.drop_prob,
                self.hetero_metadata)
        
        else:
            raise ValueError(f"{self.model_type} not one of ['hgt', 'han', 'sage']. "
                             "Please use one of the implemented models.")


        # Global pooling operator based on which one
        if(self.do_global_pool == 'mean'):
            self.pool_op = geom_nn.pool.global_mean_pool
        elif(self.do_global_pool == 'sum'):
            self.pool_op = geom_nn.pool.global_add_pool
        elif(self.do_global_pool == 'max'):
            self.pool_op = geom_nn.pool.global_max_pool
        elif(self.do_global_pool == 'none'):
            self.pool_op = None


        self.lins = nn.ModuleList()

        if(self.lin_layer_feats == -1):
            self.lin_layer_feats = [self.h_dim for _ in range(self.num_lin_layers-1)]

        # Create the readout MLP for output/classification
        if(self.num_lin_layers != 1):
            for i in range(self.num_lin_layers - 1):
                self.lins.append(nn.LazyLinear(self.lin_layer_feats[i]))
        
        self.lins.append(nn.LazyLinear(self.out_dim))

        # Add a ReLU for processing
        self.relu = nn.ReLU()

        # Include Dropout
        self.dropout = nn.Dropout(p=self.drop_prob)


    def forward(self, x_dict, edge_index_dict, x_covs, batch_num, is_sparsed, class_batch_info=None):
        
        x = self._gnn_model_forward(x_dict, edge_index_dict, batch_num, is_sparsed, class_batch_info)

        x = self._lin_model_forward(x, x_covs)

        return x
    

    def forward_with_heterodata(self, hetero_batch, **kwargs):
        # Takes in HeteroData and extracts the needed values to make a simple forward
        # When using explainers, you will need to do this separately
        # Maybe modularize this into an entirely separate function to extract and pass values
        sparsed_edge_data = 'adj_t' in hetero_batch.keys()

        graph_data_dict = hetero_batch.adj_t_dict if sparsed_edge_data else hetero_batch.edge_index_dict
        class_batch_info = hetero_batch[self.class_node].batch

        return self(x_dict=hetero_batch.x_dict, 
                    edge_index_dict=graph_data_dict,
                    x_covs=hetero_batch.covs,
                    batch_num=hetero_batch.num_graphs,
                    is_sparsed=sparsed_edge_data,
                    class_batch_info=class_batch_info,
                    **kwargs)
    
    
    def _gnn_model_forward(self, x_dict, edge_index_dict, batch_num, is_sparsed, class_batch_info=None):
        # Apply the Linear basis conversion if requested
        # (to_hetero_with_bases automatically handles this so unneeded if 'sage' was passed)

        # Modify the linear basis conversion to not modify x_dict as passed in, as this screws with explainers
        x_dict = x_dict.copy()
        edge_index_dict = edge_index_dict.copy()

        if(self.include_basis and not self.model_type == 'sage'):
            for ntype in x_dict.keys():
                x_dict[ntype] = self.relu(self.basis_lins[ntype](x_dict[ntype]))

        # Perform convolutions based on whether the data has sparsed graph info or not
        if(is_sparsed and self.model_type == 'sage'):
            raise NotImplementedError("Can't use sparsed edge data with SAGE (to_hetero_with_bases) yet...")
  
        x_dict = self.hetero_gnn(x_dict, edge_index_dict)

        # Read out the class_node features to get [n_class_nodes*n_batch, n_node_feat]
        # TODO: Consider other ways of representing this graph (noting that the number of nodes should be fixed)
        # Split along batches (if they exist - check before doing so)
        # Straightforward to get [n_batch, n_class_nodes, n_node_feat]
        x = x_dict[self.class_node]

        if(self.pool_op is not None):
            x = self.pool_op(x, batch=class_batch_info, size=batch_num)
        else:
            x = x.view(batch_num, -1)

        return x
    

    def _lin_model_to_preclass_layer_forward(self, x, x_covs):
        # Forward for the linear modules specifically
        # Helps modularize, will be used for explanations separately later

        if(self.include_covs):
            # Concatenate the individual covariates to x here
            # Consider concatenating them at the end instead though
            x = torch.concatenate([x, x_covs], dim=1)

        if(self.num_lin_layers > 1):
            for lin_layer in self.lins[:-1]:
                x = self.relu(self.dropout(lin_layer(x)))

        return x
    

    def _final_class_layer_forward(self, x):
        # Perform final classification using the node representations of the classification node
        # Consider concatenating in global features here instead of earlier
        x = self.lins[-1](x)

        if not self.explaining:
            return x
        else:
            return nn.Sigmoid()(x)


    def _lin_model_forward(self, x, x_covs):
        # Forward for the linear modules specifically
        # Helps modularize, will be used for explanations separately later

        if(self.include_covs):
            # Concatenate the individual covariates to x here
            # Consider concatenating them at the end instead though
            x = torch.concatenate([x, x_covs], dim=1)

        if(self.num_lin_layers > 1):
            for lin_layer in self.lins[:-1]:
                x = self.relu(self.dropout(lin_layer(x)))

        # Perform final classification using the node representations of the classification node
        # Consider concatenating in global features here instead of earlier
        x = self.lins[-1](x)

        if not self.explaining:
            return x
        else:
            return nn.Sigmoid()(x)
    

    def _debug_gradients(self):
        # Useful debug function that allows one to see how the gradients are being updated
        # Can be used to visualize, to some extent, the change after a loss is sent backwards
        for name, param in self.named_parameters():
            param_grad_sum = None if param.grad is None else param.grad.abs().sum()
            print(f'{name}: {param_grad_sum}')



class Homogeneous_GraphSAGE_Convs(nn.Module):
    '''
    Homogeneous GraphSAGE convolutions (to be converted to heterogeneous in a separate class)
    Just performs the convolutions - the readout and other things is handled by the heterogeneous class
    '''
    def __init__(self, input_dim, hidden_dim, num_layers, dropout):
        super().__init__()

        # Variables to store values
        self.in_dim = input_dim
        self.h_dim = hidden_dim
        self.num_layers = num_layers
        self.drop_prob = dropout

        # Model layer definitions
        self.g_convs = nn.ModuleList()

        # Add the graph convolutional layers (SAGEConv here)
        # Note that the num_classes here is actually kind of pointless since we read out at the end anyways
        # and use the node representations for the actual final class probability calculation
        # so we can probably just make in_dim -> h_dim for the full set of convolutions
        if(num_layers == 1):
            self.g_convs.append(geom_nn.SAGEConv(self.in_dim, self.h_dim))
        else:
            self.g_convs.append(geom_nn.SAGEConv(self.in_dim, self.h_dim))

            for _ in range(self.num_layers - 1):
                self.g_convs.append(geom_nn.SAGEConv(self.h_dim, self.h_dim))

        self.dropout = nn.Dropout(p=self.drop_prob)
        self.relu = nn.ReLU()


    def forward(self, x_dict, edge_index):
        # Consider adding skip-connections so that the original input information isn't lost forever after a convolution?
        # Could be a very interesting way to handle the possibility of variants having an effect beyond interactions
        # However, do not know how it would impact GNN explanation methods

        for conv_layer in self.g_convs:
            x_dict = conv_layer(x_dict, edge_index)
            # (Note, homogeneous, so can assume that x is a Tensor)
            x_dict = self.relu(x_dict)

        return x_dict


class HGT_Convs(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, hetero_metadata):
        super().__init__()

        # Variables to store values
        self.in_dim = input_dim
        self.h_dim = hidden_dim
        self.num_layers = num_layers
        self.drop_prob = dropout
        self.hetero_metadata = hetero_metadata

        # Model layer definitions
        self.g_convs = nn.ModuleList()

        # Add the graph convolutional layers (HGTConv here)
        if(num_layers == 1):
            self.g_convs.append(geom_nn.HGTConv(self.in_dim, self.h_dim, metadata=self.hetero_metadata))
        else:
            self.g_convs.append(geom_nn.HGTConv(self.in_dim, self.h_dim, metadata=self.hetero_metadata))

            for _ in range(self.num_layers - 1):
                self.g_convs.append(geom_nn.HGTConv(self.h_dim, self.h_dim, metadata=self.hetero_metadata))

        self.dropout = nn.Dropout(p=self.drop_prob)
        self.relu = nn.ReLU()

    def forward(self, x_dict, edge_index):
        n_nodes = {}

        for k, v in x_dict.items():
            n_nodes[k] = v.shape[0]
        
        for conv_layer in self.g_convs:
            x_dict = conv_layer(x_dict, edge_index)

            # Note that HGTConv will drop node types if they don't exist after a conv
            # So we reseed those features with all 0s if that does happen
            for k, v in n_nodes.items():
                x_dict[k] = self.relu(x_dict.get(k, torch.zeros(n_nodes[k], self.h_dim).to(self._get_device())))

        return x_dict
    
    def _get_device(self):
        return next(self.parameters()).device
    

class HAN_Convs(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, dropout, hetero_metadata):
        super().__init__()

        # Variables to store values
        self.in_dim = input_dim
        self.h_dim = hidden_dim
        self.num_layers = num_layers
        self.drop_prob = dropout
        self.hetero_metadata = hetero_metadata

        # Model layer definitions
        self.g_convs = nn.ModuleList()

        # Add the graph convolutional layers (HANConv here)
        if(num_layers == 1):
            self.g_convs.append(geom_nn.HANConv(self.in_dim, self.h_dim, metadata=self.hetero_metadata))
        else:
            self.g_convs.append(geom_nn.HANConv(self.in_dim, self.h_dim, metadata=self.hetero_metadata))

            for _ in range(self.num_layers - 1):
                self.g_convs.append(geom_nn.HANConv(self.h_dim, self.h_dim, metadata=self.hetero_metadata))

        self.dropout = nn.Dropout(p=self.drop_prob)
        self.relu = nn.ReLU()

    def forward(self, x_dict, edge_index):
        n_nodes = {}

        for k, v in x_dict.items():
            n_nodes[k] = v.shape[0]
        
        for conv_layer in self.g_convs:
            x_dict = conv_layer(x_dict, edge_index)

            # Note that HANConv will set the node type's features to "None" if they don't exist after a conv
            # (which is different from HGTConv, notably, which just drops that key entirely)
            # So we reseed those features with all 0s if that does happen
            for k, v in n_nodes.items():
                if(x_dict.get(k) is None):
                    x_dict[k] = torch.zeros(n_nodes[k], self.h_dim).to(self._get_device())
                else:
                    x_dict[k] = self.relu(x_dict[k])

        return x_dict
    
    def _get_device(self):
        return next(self.parameters()).device