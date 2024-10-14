from torch_geometric.explain import Explainer, CaptumExplainer
from torch_geometric.transforms import RemoveIsolatedNodes
from torch_geometric.utils import k_hop_subgraph
from captum.attr import IntegratedGradients
import torch

import os, copy
from collections import Counter, defaultdict

# Custom imports
from gnn_code.data_classes import adjt_to_edgeindex
from gnn_code.visualization_functions import visualize_hetero_explanation

from tqdm import tqdm
from functools import partialmethod

import ipdb


def explain_on_dataset(model, explain_dl, full_graph_metadata, device, 
                       topk_input=10, topk_class=10,
                       verbose=True, vis_dir='./output_explanations', do_vis=True,
                       explain_only_correct_pos=True):
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=not verbose)
    
    model.explaining = True
    os.makedirs(vis_dir, exist_ok=True)

    # Perform some explanations (basic ones; separate script for more complex ones)
    g_explainer = Explainer(
        model=model,
        algorithm=CaptumExplainer('IntegratedGradients'),
        explanation_type='model', # todo: change to phenomenon and pass target into the explainer forward
        node_mask_type='attributes',
        edge_mask_type='object',
        model_config=dict(
            mode='binary_classification',
            task_level='graph',
            return_type='probs'),
        )
    
    fc_explainer = IntegratedGradients(model._lin_model_forward, multiply_by_inputs=True)
    
    input_keyname = list(full_graph_metadata.keys())[0]
    all_top_inputs = []
    all_inputs = []

    class_keyname = model.class_node
    num_class_nodes = len(full_graph_metadata[class_keyname])
    all_top_class = []
    inverted_class_metadata = {v: k for k,v in full_graph_metadata[class_keyname].items()}

    num_pos = 0

    # TODO: Allow for passing of multiple graphs in the batch into the model
    # and then split the graphs before plotting/doing explanations
    # (should be doable by unbatching, but uncertain exactly how)

    for exp_batch_i, exp_data in enumerate(tqdm(explain_dl)):
        # Zero out model gradients
        model.zero_grad()

        exp_data = exp_data.to(device)

        if('adj_t' in exp_data.keys()):
            exp_edge_index_dict = adjt_to_edgeindex(exp_data.adj_t_dict)
        else:
            exp_edge_index_dict = exp_data.edge_index_dict
        
        g_explanation_batch = g_explainer(
            x=exp_data.x_dict,
            edge_index=exp_edge_index_dict,
            index=torch.arange(len(exp_data)),
            x_covs=exp_data.covs,
            batch_num=exp_data.num_graphs,
            is_sparsed=False
        )

        class_node_embeds = model._gnn_model_forward(exp_data.x_dict,
                                                     edge_index_dict=exp_edge_index_dict,
                                                     batch_num=exp_data.num_graphs,
                                                     is_sparsed=False)

        # Note, also attributes for covariates, but this is handled separately (currently unused)
        fcnn_class_explanation_batch, fcnn_cov_explanation_batch = fc_explainer.attribute((class_node_embeds, exp_data.covs))

        for batch_i in tqdm(range(exp_data.num_graphs), leave=False):
            batch_nodes = {k: v == batch_i for k, v in exp_data.batch_dict.items()}
            # g_orig = exp_data.subgraph(batch_nodes)
            g_explanation = g_explanation_batch.subgraph(batch_nodes)
            fcnn_class_explanation = fcnn_class_explanation_batch[batch_i]
            # fcnn_cov_explanation = fcnn_cov_explanation_batch[batch_i]

            exp_i = exp_batch_i * explain_dl.batch_size + batch_i

            curr_inputs = exp_data.var_labels[batch_i]
            graph_label = exp_data.y[batch_i].squeeze()
            pred_label = g_explanation_batch.target[batch_i].squeeze()


            tqdm.write(f"{exp_i}:\tTrue: {int(graph_label)}; Pred: {int(pred_label)}; Match?: " 
                f"{'TRUE' if int(graph_label) == int(pred_label) else 'FALSE'}")

            if(explain_only_correct_pos):
                # Skip if not positive example or if misclassified
                if(graph_label == 0):
                    tqdm.write(f"\tSkipped top {topk_input} input and {topk_class} class node listing since not a positive example\n")
                    continue
                elif(graph_label != pred_label):
                    tqdm.write(f"\tSkipped top {topk_input} input and {topk_class} class node listing since was misclassified (will explore these separately in future)\n")
                    continue
                else:
                    tqdm.write(f"\tEnumerating top {topk_input} input and {topk_class} class node listing (since was classified correctly)!\n")

            else:
                tqdm.write(f"\tEnumerating top {topk_input} input and {topk_class} class node listing!\n")

            num_pos += 1

            # Get all variants present in this graph and store them
            all_inputs.extend(curr_inputs)
            all_inputs.append('Total_explained_examples')

            # Then get the top k variants important to this graph and store them as well
            top_input_inds = g_explanation.node_mask_dict[input_keyname].abs().topk(topk_input, dim=0).indices
            inputstr_report = ''

            for var_ind in top_input_inds:
                inputstr_report += f'\t{curr_inputs[int(var_ind)]} (patient graph index {int(var_ind)})\n'
                all_top_inputs.append(curr_inputs[int(var_ind)])

            all_top_inputs.append('Total_explained_examples')

            # Get the top k class nodes important to this graph and store them as well
            # Recall that each class node actually has multiple features based on the hidden dimension
            # so we need to reshape and sum along that dimension
            fcnn_class_explanation_agg = fcnn_class_explanation.reshape(-1, num_class_nodes).sum(dim=0).abs()
            top_class_inds = fcnn_class_explanation_agg.abs().topk(topk_class, dim=0).indices
            classstr_report = ''


            for class_ind in top_class_inds:
                classstr_report += f'\t{inverted_class_metadata[int(class_ind)]} (index {int(class_ind)})\n'
                all_top_class.append(inverted_class_metadata[int(class_ind)])

            if verbose:
                tqdm.write(inputstr_report)
                tqdm.write('-')
                tqdm.write(classstr_report)
                tqdm.write('---')


            if(num_pos % 20 == 0 and verbose):
                c_input_counts = Counter(all_top_inputs)
                tqdm.write(f"\n~~~~~\nInput counts at {num_pos} examples:\n{c_input_counts}\n~~~~~")
                c_class_counts = Counter(all_top_class)
                tqdm.write(f"\n~~~~~\nClass counts at {num_pos} examples:\n{c_class_counts}\n~~~~~\n")



            # Visualizing and saving is very slow, so only do it if requested
            if(do_vis):
                vis_file = os.path.join(vis_dir, f'explain_{exp_i}.svg')

                # Add node indices to the explanation before making subgraph
                for ntype in g_explanation.node_types:
                    g_explanation[ntype].node_index = torch.arange(g_explanation[ntype].num_nodes).unsqueeze(1).to(device)

                # Threshold based on the input node features only (variants) with >0 attribution
                # and forwardprop from there to keep the relations with >0 attribution

                # Replace the classification node attribution with the ones computed
                # from the fully connected NN all the way to the end instead
                # as this is more reliable than the "initial" node features on that node
                num_class_nodes = g_explanation[model.class_node].num_nodes
                fcnn_class_explanation = fcnn_class_explanation.view(num_class_nodes, -1)
                g_explanation[model.class_node].node_mask = fcnn_class_explanation

                exp_sg = _filter_explanation(g_explanation, device, [model.class_node, input_keyname])

                sg_node_labels = {}

                for ntype in exp_sg.node_types:
                    if(ntype == input_keyname):
                        sg_node_labels[ntype] = [curr_inputs[int(i)] for i in exp_sg[ntype].node_index]
                    else:
                        inverted_gmetadata = {v: k for k,v in full_graph_metadata[ntype].items()}
                        sg_node_labels[ntype] = [inverted_gmetadata[int(i)] for i in exp_sg[ntype].node_index]
                        
                visualize_hetero_explanation(exp_sg, sg_node_labels, remove_self_loops=True, out_file=vis_file, show_fig=False)


        # Try to free up some memory once we're done with the batch
        del exp_data, g_explanation_batch, fcnn_class_explanation_batch, fcnn_cov_explanation_batch
        model.zero_grad()


    # After all the explanations are done, compute the counts
    total_top_input_counts = Counter(all_top_inputs)
    total_all_input_counts = Counter(all_inputs)
    total_top_class_counts = Counter(all_top_class)
    model.explaining = False

    total_important_input_props = {}
    # Compute proportions of topk variants over the number of times they appeared
    for k in total_top_input_counts.keys():
        total_important_input_props[k] = total_top_input_counts[k] / total_all_input_counts[k]

    # Resort by proportions
    total_important_input_props = {k: v for k, v in sorted(total_important_input_props.items(), key=lambda item: item[1], reverse=True)}


    return total_top_input_counts, total_all_input_counts, total_important_input_props, total_top_class_counts


def _filter_explanation(explanation, device='cpu', filter_node_types=None, filter_edge_types=None, rem_isolated=True, threshold=0):
    # Filters explanation by masking out zero-attribution nodes for those in filter_node_types
    # and masking out zero-attribution edges for those in filter_edge_types
    # If either is None, then all of the node/edge types will be filtered
    # (to not filter node types or edge types, pass an empty list)

    if filter_node_types is None:
        filter_node_types = explanation.node_types

    if filter_edge_types is None:
        filter_edge_types = explanation.edge_types

    node_attr_dict = explanation.collect('node_mask', True)
    edge_attr_dict = explanation.collect('edge_mask', True)

    node_mask = {}
    edge_mask = {}

    for ntype in explanation.node_types:
        if ntype in filter_node_types:
            node_mask[ntype] = node_attr_dict[ntype].sum(dim=-1) > threshold
        else:
            # All the downstream methods accept node indices instead of booleans
            # and HeteroData.subgraph does not work if given less than the number
            # of node types to subset on and the data does not live on the CPU
            # (since it generates indices that live on the CPU instead...)
            node_mask[ntype] = torch.arange(explanation[ntype].num_nodes).to(device)
    
    for etype in filter_edge_types:
        edge_mask[etype] = edge_attr_dict[etype] > threshold

    # Use explanation built-in _apply_masks function
    # with these masks to get the subgraph
    # then remove isolated nodes from the graph
    exp_sg = explanation._apply_masks(node_mask, edge_mask)

    if(rem_isolated):
        remove_isolated = RemoveIsolatedNodes()
        return remove_isolated(exp_sg)
    else:
        return exp_sg
    



def alt_explain_on_dataset(model, explain_dl, full_graph_metadata, device, 
                           topk_input=10, topk_class=10,
                           verbose=True, vis_dir='./output_explanations', do_vis=True,
                           explain_only_correct_pos=True):
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=not verbose)
    
    model.explaining = True
    os.makedirs(vis_dir, exist_ok=True)
    
    fc_explainer = IntegratedGradients(model._lin_model_forward, multiply_by_inputs=True)
    
    input_keyname = list(full_graph_metadata.keys())[0]
    pos_contrib_inputs = []
    neg_contrib_inputs = []
    all_contrib_inputs = []
    all_inputs = []

    class_keyname = model.class_node
    num_class_nodes = len(full_graph_metadata[class_keyname])
    all_top_class = []
    all_top_class_mags = defaultdict(lambda: 0)
    inverted_class_metadata = {v: k for k,v in full_graph_metadata[class_keyname].items()}

    num_pos = 0


    for exp_batch_i, exp_data in enumerate(tqdm(explain_dl)):
        # Zero out model gradients
        model.zero_grad()

        exp_data = exp_data.to(device)

        if('adj_t' in exp_data.keys()):
            exp_edge_index_dict = adjt_to_edgeindex(exp_data.adj_t_dict)
        else:
            exp_edge_index_dict = exp_data.edge_index_dict
        
        with torch.no_grad():
            class_node_embeds = model._gnn_model_forward(exp_data.x_dict,
                                                        edge_index_dict=exp_edge_index_dict,
                                                        batch_num=exp_data.num_graphs,
                                                        is_sparsed=False)

        # Note, also attributes for covariates, but this is handled separately (currently unused)
        fcnn_class_explanation_batch, fcnn_cov_explanation_batch = fc_explainer.attribute((class_node_embeds, exp_data.covs))
        exp_batch_preds = model._lin_model_forward(class_node_embeds, exp_data.covs)

        for batch_i in tqdm(range(exp_data.num_graphs), leave=False):
            batch_nodes = {k: v == batch_i for k, v in exp_data.batch_dict.items()}
            g_orig_hetero = exp_data.subgraph(batch_nodes)
            fcnn_class_explanation = fcnn_class_explanation_batch[batch_i]
            # fcnn_cov_explanation = fcnn_cov_explanation_batch[batch_i]

            exp_i = exp_batch_i * explain_dl.batch_size + batch_i

            curr_inputs = exp_data.var_labels[batch_i]
            graph_label = exp_data.y[batch_i].squeeze()
            pred_label = exp_batch_preds[batch_i].squeeze()


            tqdm.write(f"{exp_i}:\tTrue: {int(graph_label)}; Pred: {int(pred_label)}; Match?: " 
                f"{'TRUE' if int(graph_label) == int(pred_label) else 'FALSE'}")

            if(explain_only_correct_pos):
                # Skip if not positive example or if misclassified
                if(graph_label == 0):
                    tqdm.write(f"\tSkipped since not a positive example\n")
                    continue
                elif(graph_label != pred_label):
                    tqdm.write(f"\tSkipped since was misclassified (will explore these separately in future)\n")
                    continue
                else:
                    tqdm.write(f"\tEnumerating (since was classified correctly)!\n")

            else:
                tqdm.write(f"\tEnumerating (since was asked to explain everything)!\n")

            num_pos += 1

            # Get all variants present in this graph and store them
            all_inputs.extend(curr_inputs)
            all_inputs.append('Total_explained_examples')

            # Get the top k class nodes important to this graph and store them as well
            # Recall that each class node actually has multiple features based on the hidden dimension
            # so we need to reshape and sum along that dimension 
            # we also want to record whether the top class nodes were positive or negative
            fcnn_class_explanation_agg = fcnn_class_explanation.reshape(-1, num_class_nodes).sum(dim=0)
            top_class_inds = fcnn_class_explanation_agg.abs().topk(topk_class, dim=0).indices
            classstr_report = ''

            for class_ind in top_class_inds:
                top_class_name = inverted_class_metadata[int(class_ind)]
                top_class_mag = fcnn_class_explanation_agg[int(class_ind)].item()
                classstr_report += f'\t{top_class_name} (mag: {top_class_mag})\n'
                all_top_class.append(top_class_name)
            
            # We want full magnitudes for everything even if not a topk class node
            for class_name, class_mag in zip(inverted_class_metadata.values(), fcnn_class_explanation_agg.tolist()):
                all_top_class_mags[class_name] += class_mag


            # NEW/ALT METHOD
            # 2-hops to get all the variants connected to the top class nodes
            # this will be entirely count-based based on connections to the top class nodes
        
            # Need a homogeneous graph for k_hop_subgraph
            for k, v in adjt_to_edgeindex(g_orig_hetero.adj_t_dict).items():
                g_orig_hetero[k].edge_index = v
                del g_orig_hetero[k].adj_t

            g_orig = g_orig_hetero.cpu().to_homogeneous()
            input_ntype_ind = g_orig_hetero.node_types.index(input_keyname)
            class_ntype_ind = g_orig_hetero.node_types.index(class_keyname)

            input_offset = torch.argmax((g_orig.node_type == input_ntype_ind).to(dtype=torch.int)).item()
            class_offset = torch.argmax((g_orig.node_type == class_ntype_ind).to(dtype=torch.int)).item()


            n_input_max = g_orig_hetero[input_keyname].num_nodes
            connected_input_names = []

            for top_class_ind in top_class_inds:
                top_class_mag = fcnn_class_explanation_agg[top_class_ind].item()
                all_nodes_connected, _, _, _ = k_hop_subgraph(top_class_ind.item() + class_offset, 
                                                              class_ntype_ind - input_ntype_ind, 
                                                              g_orig.edge_index)
                
                input_connect_bool = (input_offset < all_nodes_connected) & (all_nodes_connected < (n_input_max + input_offset))
                input_nodes_connected = all_nodes_connected[input_connect_bool].tolist()

                mapped_names = [curr_inputs[x] for x in input_nodes_connected]

                if(top_class_mag > 0):
                    pos_contrib_inputs.extend(mapped_names)
                else:
                    neg_contrib_inputs.extend(mapped_names)

                connected_input_names.extend(mapped_names)
            

            all_contrib_inputs.extend(connected_input_names)


            # # Count and get the top ones for this graph for print reporting only
            # top_input_names = Counter(connected_input_names).most_common(topk_input)
            # inputstr_report = ''

            # for (var_name, var_count) in top_input_names:
            #     inputstr_report += f'\t{var_name} (count in this graph: {var_count})\n'


            # if verbose:
            #     tqdm.write(inputstr_report)
            #     tqdm.write('-')
            #     tqdm.write(classstr_report)
            #     tqdm.write('---')


            # if(num_pos % 20 == 0 and verbose):
            #     c_input_counts = Counter(all_contrib_inputs)
            #     tqdm.write(f"\n~~~~~\nInput counts at {num_pos} examples:\n{c_input_counts}\n~~~~~")
            #     c_class_counts = Counter(all_top_class)
            #     tqdm.write(f"\n~~~~~\nClass counts/magnitudes at {num_pos} examples:\n{c_class_counts}\n{all_top_class_mags}~~~~~\n")


        # Try to free up some memory once we're done with the batch
        del exp_data, g_orig_hetero, fcnn_class_explanation_batch, fcnn_cov_explanation_batch
        model.zero_grad()


    # After all the explanations are done, compute the counts
    total_pos_input_counts = Counter(pos_contrib_inputs)
    total_neg_input_counts = Counter(neg_contrib_inputs)
    total_abs_input_counts = Counter(all_contrib_inputs)
    total_all_input_counts = Counter(all_inputs)
    total_top_class_counts = Counter(all_top_class)
    all_top_class_mags = dict(all_top_class_mags)
    model.explaining = False

    return (total_pos_input_counts, total_neg_input_counts, total_abs_input_counts, total_all_input_counts), (total_top_class_counts, all_top_class_mags)


def _filter_explanation(explanation, device='cpu', filter_node_types=None, filter_edge_types=None, rem_isolated=True, threshold=0):
    # Filters explanation by masking out zero-attribution nodes for those in filter_node_types
    # and masking out zero-attribution edges for those in filter_edge_types
    # If either is None, then all of the node/edge types will be filtered
    # (to not filter node types or edge types, pass an empty list)

    if filter_node_types is None:
        filter_node_types = explanation.node_types

    if filter_edge_types is None:
        filter_edge_types = explanation.edge_types

    node_attr_dict = explanation.collect('node_mask', True)
    edge_attr_dict = explanation.collect('edge_mask', True)

    node_mask = {}
    edge_mask = {}

    for ntype in explanation.node_types:
        if ntype in filter_node_types:
            node_mask[ntype] = node_attr_dict[ntype].sum(dim=-1) > threshold
        else:
            # All the downstream methods accept node indices instead of booleans
            # and HeteroData.subgraph does not work if given less than the number
            # of node types to subset on and the data does not live on the CPU
            # (since it generates indices that live on the CPU instead...)
            node_mask[ntype] = torch.arange(explanation[ntype].num_nodes).to(device)
    
    for etype in filter_edge_types:
        edge_mask[etype] = edge_attr_dict[etype] > threshold

    # Use explanation built-in _apply_masks function
    # with these masks to get the subgraph
    # then remove isolated nodes from the graph
    exp_sg = explanation._apply_masks(node_mask, edge_mask)

    if(rem_isolated):
        remove_isolated = RemoveIsolatedNodes()
        return remove_isolated(exp_sg)
    else:
        return exp_sg
    
