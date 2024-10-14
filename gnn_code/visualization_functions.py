import torch

import ipdb

import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Use Netgraph if installed and has multigraph support
try:
    from netgraph import MultiGraph
    NG_AVAIL = True
except:
    NG_AVAIL = False


def visualize_hetero_explanation(explanation, node_labels=None, remove_self_loops=False, 
                                 out_file=None, show_fig=False):

    explanation = explanation.detach().to('cpu')

    node_weights = {k: v.abs().sum(dim=1) for k,v in explanation.node_mask_dict.items()}
    node_weights = {k: (v - v.min()) / (v.max() - v.min()) for k,v in node_weights.items()}

    # Need to handle edge types that have essentially been removed
    # Most likely by removing those edges entirely from the heterodata
    edge_weights = {}
    for k,v in explanation.edge_mask_dict.items():
        if v.numel() == 0:
            edge_weights[k] = torch.empty(0)
        else:
            edge_weights[k] = (v - v.min()) / (v.max() - v.min())

    # Need a size to be realistically visible that we can scale from
    base_node_size = max(5, 300.0 / explanation.num_nodes)
    base_edge_size = 1.0
    base_font_size = max(1, int(base_node_size/25))

    # Determine whether using Netgraph is reasonable (it is extremely slow on large graphs)
    no_use_NG = explanation.num_nodes > 100

    visualize_heterograph(explanation, node_labels, node_weights, edge_weights, 
                          base_node_size, base_edge_size, base_font_size,
                          remove_self_loops, no_use_NG, 
                          out_file, show_fig)


def visualize_heterograph(hdata, node_labels=None, node_weights=None, edge_weights=None, 
                          base_node_size=10, base_edge_size=0.5, base_font_size=5,
                          remove_self_loops=True, no_use_NG=True,
                          out_file=None, show_fig=False):
    G = nx.MultiDiGraph()

    # Create a homogeneous version of the graph
    # Does the node labeling and such for us
    # But we do need to maintain the key order in case
    # the other arguments were in a passed order
    homo_hdata = hdata.to_homogeneous()


    G.add_nodes_from(range(homo_hdata.num_nodes))
    nx.set_node_attributes(G, {i: float(v) for i,v in enumerate(homo_hdata.node_type)}, 'node_type')

    node_weights = _construct_networkx_attribute(node_weights, homo_hdata.num_nodes, hdata.node_types, base_node_size)
    nx.set_node_attributes(G, {i: float(v) for i,v in enumerate(node_weights)}, 'node_weight')


    node_labels = _construct_networkx_attribute(node_labels, homo_hdata.num_nodes, hdata.node_types, '')
    nx.set_node_attributes(G, {i: v for i,v in enumerate(node_labels)}, 'node_label')


    # Setting edge attributes is far more annoying
    # So we do loop over the edges for this one
    edge_list = homo_hdata.edge_index.t().tolist()
    edge_weights = _construct_networkx_attribute(edge_weights, homo_hdata.num_edges, hdata.edge_types, base_edge_size)
    edge_types = homo_hdata.edge_type

    for edge_vals, edge_weight, edge_type in zip(edge_list, edge_weights, edge_types):
        G.add_edge(edge_vals[0], edge_vals[1], edge_weight=float(edge_weight), edge_type=float(edge_type))
    
    
    # Get graph positions
    # TODO: maybe add some random vertical adjustment in a layer or
    # arc the different node types instead of a straight line
    # to make it possible to show within-layer interactions
    # in a meaningful manner (right now, they all intersect)
    pos = nx.multipartite_layout(G, subset_key="node_type", align='horizontal')

    # Offset nodes to spread out as much as possible horizontally
    # Loop over each layer and get the nodes in that layer
    node_pos_scales = []

    for layer_num in range(len(homo_hdata.node_type.unique())):
        layer_inds = (homo_hdata.node_type == layer_num).nonzero().squeeze()
        layer_pos_scale = 1.0 / pos[int(layer_inds[0])][0]
        node_pos_scales += [layer_pos_scale] * len(layer_inds)

    pos = [np.array([pos[i][0] * node_pos_scale, pos[i][1]]) 
           for i, node_pos_scale in enumerate(node_pos_scales)]


    if(remove_self_loops):
        G.remove_edges_from(list(nx.selfloop_edges(G)))


    node_color_args = {'node_color': list(nx.get_node_attributes(G,'node_type').values()),
                  'vmin': 0,
                  'vmax': len(hdata.node_types),
                  'cmap': plt.cm.get_cmap('Pastel1')}
    
    node_size_args = {'node_size': list(nx.get_node_attributes(G,'node_weight').values())}
    
    node_label_args = {'labels': nx.get_node_attributes(G,'node_label'),
                       'font_size': base_font_size,
                       'font_weight': 'ultralight',
                       'alpha': 0.5}
    
    edge_color_args = {'edge_color': list(nx.get_edge_attributes(G,'edge_type').values()),
                    'edge_vmin': 0,
                    'edge_vmax': len(hdata.edge_types),
                    'edge_cmap': plt.cm.get_cmap('Pastel2')}
    
    edge_size_args = {'width': list(nx.get_edge_attributes(G,'edge_weight').values()),
                       'arrowsize': base_edge_size*1.2,
                       'node_size': base_node_size}



    plt.figure(figsize=(96,32))

    if(NG_AVAIL and not no_use_NG):
        # Netgraph "handles" node and edge sizing (?)

        ncolor = _acquire_colormap_values(nx.get_node_attributes(G,'node_type'), 
                                          node_color_args['cmap'],
                                          node_color_args['vmin'], node_color_args['vmax'])
        
        ecolor = _acquire_colormap_values(nx.get_edge_attributes(G, 'edge_type'),
                                          edge_color_args['edge_cmap'],
                                          edge_color_args['edge_vmin'],
                                          edge_color_args['edge_vmax'])


        MultiGraph(G,
                   node_layout=pos,
                   node_labels=node_label_args['labels'],
                   edge_layout='curved',
                   edge_layout_kwargs=dict(bundle_parallel_edges=False),
                   arrows=True,
                   edge_color=ecolor,
                   node_color=ncolor,
                   )
        
    else:
        nx.draw_networkx_nodes(G, pos, **node_color_args, **node_size_args)
        nx.draw_networkx_edges(G, pos, **edge_color_args, **edge_size_args, connectionstyle='arc3')
        label_text_data = nx.draw_networkx_labels(G, pos, **node_label_args)

        for _, label_text in label_text_data.items():
            # label_text.set_weight('ultralight')
            # label_text.set_alpha(0.5)
            label_text.set_rotation('vertical')


    if(out_file is not None):
        plt.savefig(out_file, dpi=80)

    if(show_fig):
        plt.show()

    plt.close()



def _dict_to_concatenated_list(in_dict, key_order=None, base_value=1):
    if(isinstance(base_value, str)):
        base_value = 1

    if(key_order is not None):
        out_list = []

        for key in key_order:
            out_list.append(np.array(in_dict[key]))
    
    else:
        out_list = [np.array(key_list) for key_list in in_dict.values()]
    
    out_list = [key_list * base_value if key_list.dtype.kind not in ['U', 'S'] else key_list
                for key_list in out_list]
    
    out_list = np.hstack(out_list).tolist()
    
    return out_list


def _construct_networkx_attribute(in_attr_vals, num_vals, key_order=None, base_value=1):
    if(in_attr_vals is not None):
        if isinstance(in_attr_vals, dict):
            out_list = _dict_to_concatenated_list(in_attr_vals, key_order, base_value)

        assert len(out_list) == num_vals
    else:
        out_list = [base_value for _ in range(num_vals)]

    return out_list


def _acquire_colormap_values(in_dict, cmap, vmin=None, vmax=None):
    # Heavily inspired by _get_color from netgraph

    keys = in_dict.keys()
    values = np.array(list(in_dict.values()), dtype=float)

    # apply vmin, vmax
    if(vmin is not None and vmax is not None):
        values = (values - vmin) / (vmax - vmin)

    # convert value to color
    mapper = plt.cm.ScalarMappable(cmap=cmap)
    mapper.set_clim(vmin=0., vmax=1.)
    colors = mapper.to_rgba(values)

    return {key: color for (key, color) in zip(keys, colors)}
