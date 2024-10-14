import torch

from torch_geometric.transforms import ToSparseTensor
from torch_geometric.data import HeteroData
from torch_geometric.utils import degree

import numpy as np
import re, os
import collections

import ipdb


def custom_train_test_split(data, train_prop=0.7, seed=9):

    n_total = len(data)
    n_train = int(np.around(train_prop * n_total))

    rng = torch.Generator().manual_seed(seed)

    return torch.utils.data.random_split(data, [n_train, n_total-n_train], rng)


def custom_load_genomics(adj_mat_dict, indiv_data_dict, do_filter=True):
    # Graph-specific data (same for every subject)
    edge_index = adj_mat_dict['edge_indices']
    graph_metadata = adj_mat_dict['node_mappings']

    # Individual data (different for each subject)
    indiv_data = indiv_data_dict['data']
    indiv_labels = indiv_data_dict['target']
    indiv_metadata = indiv_data_dict['SubjID']
    indiv_covs = indiv_data_dict.get('covariates')

    # Filtering if asked for, based on target (or covariates) being nan
    if(do_filter):
        prune_indiv_bool = np.isnan(indiv_labels)
        if(indiv_covs is not None):
            prune_indiv_bool_covs = (np.isnan(indiv_covs).sum(axis=1) > 0)
            prune_indiv_bool = prune_indiv_bool | prune_indiv_bool_covs

        prune_indiv_indices = torch.from_numpy(np.flatnonzero(~prune_indiv_bool))

        # Use torch.index_select to select indices of the sparse tensor
        # Can't use standard indexing (i.e. indiv_data[~prune_indiv_bool] as is unimplemented)
        indiv_data = torch.index_select(indiv_data, 0, prune_indiv_indices)
        indiv_labels = torch.tensor(indiv_labels)[~prune_indiv_bool]
        indiv_metadata = np.array(indiv_metadata)[~prune_indiv_bool]
        indiv_covs = torch.tensor(indiv_covs)[~prune_indiv_bool]
        
    else:
        indiv_labels = torch.tensor(indiv_labels)
        indiv_metadata = np.array(indiv_metadata)
        indiv_covs = torch.tensor(indiv_covs)

    # Convert everything to the right datatypes
    indiv_data = indiv_data.float()
    indiv_covs = indiv_covs.float()
    indiv_labels = indiv_labels.long()

    # Check node information for weird characters
    fix_node_type_names = {}
    n_fix = 0
    for ntype in graph_metadata.keys():
        ntype_parse = re.sub('[^a-zA-Z0-9]', '_', ntype)

        if(ntype_parse != ntype):
            fix_node_type_names[ntype] = ntype_parse
            n_fix += 1
        else:
            fix_node_type_names[ntype] = ntype

    # If any weird characters found in the node names, then time to fix them

    # Fix in graph metadata
    if(n_fix > 0):
        for oldkey, newkey in fix_node_type_names.items():
            graph_metadata[newkey] = graph_metadata.pop(oldkey)
    
    # Fix in edge data (only fixing node names for now)
    # If we need to fix edge names, we'll make this a function in future
    fix_edge_tuples = {}
    e_fix = 0
    for etuple in edge_index.keys():
        new_tuple = tuple([fix_node_type_names[etuple[0]], 
                           etuple[1],
                           fix_node_type_names[etuple[2]]])
        
        if(etuple != new_tuple):
            fix_edge_tuples[etuple] = new_tuple
            e_fix += 1
        else:
            fix_edge_tuples[etuple] = etuple

    if(e_fix > 0):
        for oldkey, newkey in fix_edge_tuples.items():
            edge_index[newkey] = edge_index.pop(oldkey)

    # Reparse the data to appropriate shapes and formats
    if(len(indiv_data.shape) < 3):
        indiv_data = indiv_data.unsqueeze(-1)

    all_ntypes = list(graph_metadata.keys())

    if(not isinstance(indiv_data, collections.abc.Mapping)):
        n_samp = indiv_data.shape[0]
        n_feat = indiv_data.shape[-1]
        indiv_data = {all_ntypes[0]: indiv_data}

        for other_ntype in all_ntypes[1:]:
            num_nodes_ntype = len(graph_metadata[other_ntype])
            indiv_data[other_ntype] = torch.zeros(n_samp, num_nodes_ntype, n_feat).to_sparse()

    return edge_index, graph_metadata, indiv_data, indiv_covs, indiv_labels, indiv_metadata
    

def select_index_dict(in_dict_of_tensors, dim, idx, squeeze_first=True):
    out_dict = {}

    for key, value in in_dict_of_tensors.items():
        value = torch.index_select(value, dim, idx)

        if(squeeze_first):
            if(value.is_sparse):
                out_dict[key] = torch.sparse.sum(value, [0])
            else:
                out_dict[key] = value.squeeze(0)

    return out_dict


def densify_sparsetensor_dict(in_dict_of_tensors):
    out_dict = {}

    for key, value in in_dict_of_tensors.items():
        if(value.is_sparse):
            out_dict[key] = value.to_dense()
        else:
            out_dict[key] = value

    return out_dict


def adjt_to_edgeindex(adj_t_dict):
    edge_index_dict = dict()

    for k, adj_t in adj_t_dict.items():
        row, col = adj_t.t().to_sparse_coo().coalesce().indices()
        edge_index_dict[k] = torch.stack([row, col], dim=0)

    return edge_index_dict




class VariantGenePathwayDataset(torch.utils.data.Dataset):
    def __init__(self, adj_mat_dict, indiv_data_dict, 
                 make_sparsed_edges=True, make_sparsed_nodes=True,
                 warm_start_data=None, save_every=1024,
                 add_node_indegree_data=True):
        # Use custom_load_genomics function to process
        self.edge_index, self.graph_metadata, self.indiv_data, self.indiv_covs, self.indiv_labels, self.indiv_metadata = \
            custom_load_genomics(adj_mat_dict, indiv_data_dict)
        
        self.make_sparsed_edges = make_sparsed_edges
        self.make_sparsed_nodes = make_sparsed_nodes

        if(add_node_indegree_data):
            # Add indegree information for all non-variant nodes
            # Note this will add multiple features, one for each incoming edge type
            # As such, different node types may have different numbers of features in the end
            # and this will need to be handled by the GNN forward
            non_var_ntypes = list(self.graph_metadata.keys())[1:]
            for ntype in non_var_ntypes:
                ntype_vals = None
                ntype_num = len(self.graph_metadata[ntype])
                incoming_edge_types = [triplet for triplet in self.edge_index.keys() if triplet[2] == ntype]

                for in_etype in incoming_edge_types:
                    in_etype_idx = self.edge_index[in_etype][1]
                    etype_ntype_degrees = degree(in_etype_idx, ntype_num).unsqueeze(1)
                    etype_ntype_degrees = etype_ntype_degrees / etype_ntype_degrees.max()

                    if(ntype_vals is None):
                        ntype_vals = etype_ntype_degrees
                    else:
                        ntype_vals = torch.cat([ntype_vals, etype_ntype_degrees], dim=-1)

                # Need to expand this data across all of the samples
                # Expand instead of repeat doesn't allocate more memory
                self.indiv_data[ntype] = ntype_vals.unsqueeze(0).expand(len(self),-1,-1)
            
        
        # Create an inverted dict for faster lookups of var labels
        self.inverted_var_dict = {v: k for k,v in self.graph_metadata[list(self.graph_metadata.keys())[0]].items()}

        # Values to store processed heterodata (and load from file if provided)
        self.curr_heterodata = {}
        self.warm_start_file = warm_start_data
        self.save_every = save_every
        self.last_saved = 0

        if(warm_start_data is None):
            self.save_data = False

        else:
            self.save_data = True
            if(os.path.isfile(warm_start_data)):
                self.curr_heterodata = torch.load(warm_start_data)
                self.last_saved = len(self.curr_heterodata)


    def __getitem__(self, idx):
        # First check if the data already exists in our stored dict
        # If so, just use it instead of processing it again
        exist_data = self.curr_heterodata.get(idx)
        if(exist_data is not None):
            return exist_data
        # If the data doesn't exist, then we need to process it as below

        # index_select only works for tensor indices
        if(not torch.is_tensor(idx)):
            idx_t = torch.tensor(idx, dtype=torch.long)

        # index_select breaks if the tensor is 0D (aka has one element)
        # There is probably a better way to handle this but this works for now
        if(idx_t.numel() == 1):
            idx_t = idx_t.unsqueeze(dim=0)
            squeeze_first = True
        
        sel_data = select_index_dict(self.indiv_data, 0, idx_t, squeeze_first=squeeze_first)
        sel_data = densify_sparsetensor_dict(sel_data)
        
        # Create HeteroData object now for this data
        out_heterodata = CustomHeteroData()

        # Populate the heterodata with node features and edge indices (and num_nodes)
        for ntype, nfeats in sel_data.items():
            out_heterodata[ntype].x = nfeats

        for tripletdef, eindexes in self.edge_index.items():
            out_heterodata[tripletdef].edge_index = eindexes

        # Drop variant nodes in the graph with feature 0
        # and store a list of their original IDs
        if(self.make_sparsed_nodes):
            # Get the variants with 0 feature/count
            variant_key = list(self.graph_metadata.keys())[0]
            v_data = out_heterodata[variant_key].x
            keep_v = v_data.nonzero(as_tuple=True)[0]
            out_heterodata = out_heterodata.subgraph({variant_key: keep_v})
            
            v_labels = list(map(self.inverted_var_dict.get, keep_v.tolist()))
            out_heterodata.var_labels = v_labels

        # Convert edge index to sparsed
        if(self.make_sparsed_edges):
            st_tf = ToSparseTensor()
            out_heterodata = st_tf(out_heterodata)

        out_heterodata.y = self.indiv_labels[idx_t]
        out_heterodata.covs = self.indiv_covs[idx_t]

        # Long-term storage to prevent needing to reget data
        # if this idx is called again in the future
        self.curr_heterodata[idx] = out_heterodata
        if(self.save_data and (len(self.curr_heterodata) - self.last_saved) % self.save_every == 0):
            self._dump_data_to_file()

        return out_heterodata
    
    def __len__(self):
        return self.indiv_labels.size(0)
    
    def get_hetero_metadata(self):
        parsed_hetero_metadata = (list(self.graph_metadata.keys()), 
                                  list(self.edge_index.keys()))

        return parsed_hetero_metadata
    
    def get_shared_edge_index(self):
        return self.edge_index
    
    def get_graph_metadata(self):
        return self.graph_metadata
    
    def _dump_data_to_file(self):
        torch.save(self.curr_heterodata, self.warm_start_file)
        self.last_saved = len(self.curr_heterodata)
    


# Need the below because of issues with batching on sparsed data: https://github.com/pyg-team/pytorch_geometric/issues/8355
# Related to __cat_dim__ not returning the correct value for torch.sparse tensors due to mistake in the check for sparse data 
# (specifically, was still checking for SparseTensor)
# I've submitted a PR to fix this but will continue to be working on the release version rather than on master, so this is needed until a new release

from typing import Any, Optional, Union
from torch_geometric.data.storage import EdgeStorage, NodeStorage
from torch_geometric.utils.sparse import is_sparse


NodeOrEdgeStorage = Union[NodeStorage, EdgeStorage]

class CustomHeteroData(HeteroData):
    def __cat_dim__(self, key: str, value: Any,
                    store: Optional[NodeOrEdgeStorage] = None, *args,
                    **kwargs) -> Any:
        
        if is_sparse(value) and 'adj' in key:
            return (0, 1)
        elif isinstance(store, EdgeStorage) and 'index' in key:
            return -1
        return 0