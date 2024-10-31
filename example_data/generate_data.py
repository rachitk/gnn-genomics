import torch
import numpy as np

### Generate some toy data
### matching the expected structure of the dataset as input

# Fix the random seed (to try to maximize reproducibility)
torch.manual_seed(999)

# Define the number of nodes and edges
# as well as their types
n_samples = 1000
n_variants = 2000
n_genes = 500
n_groups = 100

var_type_name = '#Uploaded_variation'
gene_type_name = 'Gene'
group_type_name = 'group'

variant_to_gene_types = ['splice_donor_variant', 'missense_variant', 'splice_acceptor_variant', 
                         'stop_gained', 'start_lost', 'stop_lost', 'inframe_deletion', 
                         'frameshift_variant', 'inframe_insertion']

gene_to_group_types = ['in']

## graph_structure.pt
## Stores a nested Python dictionary with the following key, value pairs:
# [node_mappings] contains a Python dictionary:
#  - keys are strings representing node types (e.g. '#Uploaded variation', 'Gene', 'group')
#  - values are dictionaries mapping node identifiers to integer indices, 0-indexed
#      - e.g. node_mappings['Gene']['ENSG00000123373'] = 0
# [edge_indices] contains a Python dictionary:
#  - keys are tuples representing (source node type, edge type, destination node type); e.g. ('#Uploaded_variation', 'splice_donor_variant', 'Gene') or ('Gene', 'in', 'group')
#  - values are torch tensors of shape [2, n_edges] representing edges as indices of source nodes to indices of destination nodes


# Make fake node mappings
var_map = {f'rs{i}': i for i in range(n_variants)}
gene_map = {f'ENSG{i}': i for i in range(n_genes)}
group_map = {f'group{i}': i for i in range(n_groups)}

node_mappings = {var_type_name: var_map, 
                 gene_type_name: gene_map, 
                 group_type_name: group_map}


# Make fake edge indices 
etypes = []
src_nodes = []
dst_nodes = []

# (randomly connect variants to 1-2 genes each, picking a random edge type)
var_num_edges = torch.randint(1, 3, (n_variants,))

for i in range(n_variants):
    for j in range(var_num_edges[i]):
        etype = variant_to_gene_types[torch.randint(0, len(variant_to_gene_types), (1,)).item()]
        tuple_etype = (var_type_name, etype, gene_type_name)
        etypes.append(tuple_etype)
        src_nodes.append(i)
        dst_nodes.append(torch.randint(0, n_genes, (1,)).item())


# (randomly connect genes to 1-2 groups each, picking a random edge type)
gene_num_edges = torch.randint(1, 3, (n_genes,))

for i in range(n_genes):
    for j in range(gene_num_edges[i]):
        etype = gene_to_group_types[torch.randint(0, len(gene_to_group_types), (1,)).item()]
        tuple_etype = (gene_type_name, etype, group_type_name)
        etypes.append(tuple_etype)
        src_nodes.append(i)
        dst_nodes.append(torch.randint(0, n_groups, (1,)).item())


# Make edge index as dict with keys as tuples and values as torch tensors
edge_indices = {}
unique_etypes = list(set(etypes))

for etype in unique_etypes:
    src = torch.tensor([src_nodes[i] for i in range(len(etypes)) if etypes[i] == etype])
    dst = torch.tensor([dst_nodes[i] for i in range(len(etypes)) if etypes[i] == etype])
    edge_indices[etype] = torch.stack([src, dst])


# Make final dict to save
graph_structure = {'node_mappings': node_mappings, 
                   'edge_indices': edge_indices}

# Save the graph structure
torch.save(graph_structure, 'example_graph_structure.pt')

print('Graph structure generation complete! File saved as example_graph_structure.pt')


## collated_tensors_withPheno_noSepAPOE.pt
## Stores a Python dictionary with a variety of items
# [SubjID] contains a Python list with each element being a string representing a subject identifier
# [VarID] contains a Python list with each element being a string representing a variant identifier (this is identical to the keys of graph_structure['node_mappings']['#Uploaded_variation'])
# [data] contains a SPARSE torch tensor of shape [n_samples, n_variants] representing the genotype data. This is typically extremely sparse (on the order of 0.4% sparsity in the ADSP data)
# [target] contains a numpy array of shape [n_samples] representing the target variable (in the case of ADSP, this was AD case/control status). Note that this can contain NaNs, which are dropped later.
# [covariates] contains a numpy array of shape [n_samples, n_covariates] representing the covariates. Note that this can contain NaNs, which are dropped later.

# Make fake subject IDs
subj_ids = [f'Subj{i}' for i in range(n_samples)]

# Variant IDs are a copy of the keys of the node mappings
var_ids = list(node_mappings[var_type_name].keys())

# Make fake genotype data (note, we want high levels of sparsity)
geno_data_dense = (torch.rand(n_samples, n_variants) < 0.004).long()
geno_data = geno_data_dense.to_sparse()

# Make fake target variable (binary)
# Make it depend on the first 100 variants 
# (no good way to force it to depend on some groups in a biologically rigorous manner, 
# but for this toy example, it doesn't matter)
target = geno_data_dense[:,:100].sum(dim=1)
target = (target > 0).double().numpy()

# Make fake covariates
# Sex and age (unscaled), sex = 0/1, age = 50-90
covariates = np.zeros((n_samples, 2), dtype=np.double)
covariates[:,0] = torch.randint(0, 2, (n_samples,))
covariates[:,1] = torch.randint(50, 91, (n_samples,))

# Corrupt the target and covariates by adding some NaNs at a 0.001 rate
nan_mask = (torch.rand(n_samples) < 0.001).long()
target[nan_mask] = np.nan
covariates[nan_mask, :] = np.nan

# Make final dict to save
collated_tensors = {'SubjID': subj_ids, 
                   'VarID': var_ids, 
                   'data': geno_data, 
                   'target': target, 
                   'covariates': covariates}

# Save the collated tensors
torch.save(collated_tensors, 'example_collated_tensors_withPheno_noSepAPOE.pt')

print('Input data generation complete! File saved as example_collated_tensors_withPheno_noSepAPOE.pt')
