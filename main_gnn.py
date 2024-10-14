import argparse

import torch
import torch.nn as nn
import torch_geometric
import numpy as np

from collections import Counter

import random
import os, csv
import warnings

from functools import partialmethod


# Custom classes below
import gnn_code.gnn_model as gnn_model
import gnn_code.train_eval_functions as train_eval_functions
import gnn_code.explanation_functions as explanation_functions
# import gnn_code.data_classes as data_classes
from gnn_code.data_classes import custom_train_test_split
from gnn_code.data_classes import VariantGenePathwayDataset

# Debugging
import ipdb
from tqdm import tqdm

# Suppress an annoying warning related to sparse tensors
warnings.filterwarnings('ignore', '.*Sparse CSR tensor support is in beta state.*')



def main(args):
    
    # Set the random seed
    # Using builtin torch_geometric function
    torch_geometric.seed_everything(args.seed)
    # np.random.seed(args.seed)
    # torch.manual_seed(args.seed)
    # random.seed(args.seed)


    # Set the device (if CUDA is available and user has not forced CPU)
    # Note that if there are multiple GPUs and this is run using torchrun
    # the device will be overridden by the DistributedDataParallel setup
    multi_gpu_use = False

    if torch.cuda.is_available() and not args.force_cpu:
        if torch.cuda.device_count() > 1:
            if(not args.enable_multigpu):
                device = torch.device("cuda")
                print('{} GPUs detected, but --enable-multigpu not passed, so model will not use torch DistributedDataParallel with all the GPUs.'.format(torch.cuda.device_count()))
            elif(not torch.distributed.is_torchelastic_launched()):
                device = torch.device("cuda")
                print('{} GPUs detected, but this script was not run using torchrun, so model will not use torch DistributedDataParallel with all the GPUs.'.format(torch.cuda.device_count()))
            else:
                print('{} GPUs detected; model will use torch DistributedDataParallel with all the GPUs for batch processing.'.format(torch.cuda.device_count()))
                torch.distributed.init_process_group(backend="nccl")
                device = int(os.environ["LOCAL_RANK"])
        else:
            device = torch.device("cuda")
            print("\tUsing one GPU for training...")
    else:
        device = torch.device("cpu")
        print("\tUsing CPU for training...")



    # Disable TQDM progress bar if user does not explicitly request it
    # Also disable if device ID not 0 if dispatching distributed processing
    if(not multi_gpu_use):
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=not args.enable_progress_bar)
    else:
        tqdm.__init__ = partialmethod(tqdm.__init__, disable=not args.enable_progress_bar and device!=0)


    # Load data and identify things like number of features and such
    # data loading of tensors
    edge_index_dict = torch.load(args.input_data_adj)
    indiv_data_dict = torch.load(args.input_data_features)

    if(args.data_reload_prefix):
        reload_data_file = f'{args.data_reload_prefix}.pt'
    else:
        reload_data_file = None

    print("\tInitializing dataset (and loading if data exists)...")

    vgp_adsp_dataset = VariantGenePathwayDataset(edge_index_dict, indiv_data_dict, 
                                                 make_sparsed_edges=args.make_sparsed_edges, 
                                                 make_sparsed_nodes=args.make_sparsed_nodes,
                                                 warm_start_data=reload_data_file,
                                                 save_every=1024,
                                                 add_node_indegree_data=args.add_node_indegree_as_feat)

    print("\tFinished initializing dataset!")


    # Set the data to be used (useful line for swapping Fake and ADSP)
    data = vgp_adsp_dataset

    num_classes = len(data.indiv_labels.unique())
    hetero_nodeedge_metadata = vgp_adsp_dataset.get_hetero_metadata()
    full_graph_metadata = vgp_adsp_dataset.get_graph_metadata()

    coerce_binary = False

    # Split to get training data first
    train_data, test_data = custom_train_test_split(data, args.train_prop, seed=args.seed)

    # Split again using the proportion of testing to validation data
    # This gets us the proper test and validation proportion we want
    test_data, val_data = custom_train_test_split(test_data, args.test_prop/(args.test_prop+args.val_prop), seed=args.seed)

    if(args.do_class_weighting):
        # Get counts of classes for each of the labels (in the training data only)
        print("Getting counts of each class to try to weight infrequent classes highly...")
        lb_arr = data.indiv_labels.numpy()
        lb_counts = Counter(lb_arr)

        lb_count_all = np.array([lb_counts.get(i, 1) for i in range(num_classes)])
        lb_weight_all = torch.from_numpy(lb_count_all.sum() / lb_count_all).to(dtype=torch.float).to(device)
    else:
        print("Will not weight classes at all...")
        # Pass weight of all ones (aka no weights, basically)
        lb_weight_all = torch.ones(num_classes)

    lb_weight_all = lb_weight_all / lb_weight_all.sum()

    # Coerce binary if this is binary classification
    if(num_classes == 2):
        num_classes = 1
        coerce_binary = True
        lb_weight_all = lb_weight_all[1] / lb_weight_all[0]

    # Create dataloaders
    train_dl = torch_geometric.loader.DataLoader(train_data, shuffle=True, batch_size=args.batch_size, drop_last=False)
    val_dl = torch_geometric.loader.DataLoader(val_data, shuffle=True, batch_size=args.batch_size, drop_last=False)
    test_dl = torch_geometric.loader.DataLoader(test_data, shuffle=False, batch_size=args.batch_size, drop_last=False)


    # Define the model input dimensions based on either 
    # a given value or an assumption if not given
    if(args.basis_dim is None):
        input_dim = 1
        input_dim_basis = False
    else:
        input_dim = args.basis_dim
        input_dim_basis = True

    model_kwargs = dict(input_dim=input_dim,
            hidden_dim=args.hidden_dim,
            num_classes=num_classes,
            num_conv_layers=args.num_conv_layers,
            num_lin_layers=args.num_lin_layers,
            hetero_metadata=hetero_nodeedge_metadata,
            graph_metadata=full_graph_metadata,
            lin_layer_feats=args.lin_layer_feats,
            class_node='group',
            dropout=args.dropout_prob,
            include_basis=input_dim_basis,
            include_covs=not args.no_include_covs,
            do_global_pool=args.global_pool_type,
            model_type=args.model_type)

    # Construct the model
    model = gnn_model.HeterogeneousGraphClassifier(**model_kwargs).to(device)

    print("Initializing model parameters now...")
    
    # Call into the data with a dummy batch to initialize the Lazy Modules
    with torch.no_grad():
        dummy_batch = next(iter(train_dl))
        
        _ = model.forward_with_heterodata(dummy_batch.to(device))

        # Report model complexity and trainable parameters for each layer
        num_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print("\nThe model has {} trainable parameters in total".format(num_total_params))

        for name, param in model.named_parameters():
            if not param.requires_grad: 
                n_params = 0
            else:
                n_params = param.numel()

            print("\t{} has {} trainable parameters".format(name, n_params))
        
        print()

        del dummy_batch


    # Dispatch distributed data if we have multiple GPUs being used
    # if(multi_gpu_use):
    #     model = DistributedDataParallel(model, device_ids=[i], output_device=i)


    # Load the model if the user says they want to load a particular model as a warm-start
    if args.warm_start_model is not None:
        print(f"Loading model from {args.warm_start_model}\n")
        model.load_state_dict(torch.load(args.warm_start_model, map_location=device))

    # if not args.no_compile:
    #     model = torch_geometric.compile(model, dynamic=True, fullgraph=True)

    if not args.skip_training:
        print(f"Training model now...\n")

        
        try:
            # Train the model and save the trained model
            model = train_eval_functions.training_loop(
                model=model, 
                device=device, 
                lr=args.lr, 
                epochs=args.epochs, 
                train_data=train_dl,
                val_data=val_dl,
                label_weights=lb_weight_all,
                verbose=args.enable_progress_bar,
                save_epochs=1,
                save_location=args.out_dir,
                save_prefix=args.model_prefix,
                val_epochs=args.val_epochs,
                coerce_binary=coerce_binary,
                opt_type=args.optimizer,
                early_stop_thresh=args.early_stop)

            torch.save(model.state_dict(), os.path.join(args.out_dir, '{}.pt'.format(args.model_prefix)))

        except KeyboardInterrupt:
            try:
                tqdm._instances.clear() #try to fix TQDM issue with interrupting
            except:
                pass

            print("\nTraining interrupted by user. Model will be saved as is and a debug terminal opened.\n")
            torch.save(model.state_dict(), os.path.join(args.out_dir, '{}_interrupted.pt'.format(args.model_prefix)))
            ipdb.set_trace()


        if(args.downstream_on_best):
            print(f"Loading the model with the best validation loss now for downstream evaluation and explanation...\n")
            model.load_state_dict(torch.load(os.path.join(args.out_dir, '{}_best_val_loss.pt'.format(args.model_prefix))))


    if not args.skip_eval:
        print(f"Evaluating model now...\n")
        metrics_dir = os.path.join(args.out_dir, 'metrics_dir')
        out_metrics_file = os.path.join(metrics_dir, 'metrics.csv')
        os.makedirs(metrics_dir, exist_ok=True)

        # Define loss function - currently same used in the original training loop 
        # (may make these selectable in the future?)
        if(coerce_binary):
            # Binary loss function
            loss_func = nn.BCEWithLogitsLoss(pos_weight=lb_weight_all.to(device)).to(device)
        else:
            # Multiclass loss function
            loss_func = nn.CrossEntropyLoss(weight=lb_weight_all.to(device)).to(device)

        # Evaluate the model on the holdout test dataset
        metrics, fig, embeds_and_labels = train_eval_functions.model_evaluation(model, device, test_dl, loss_func, coerce_binary, 
                                                                               args.do_embed_vis, args.embed_vis_method, args.use_pathway_embeds,
                                                                               verbose=args.enable_progress_bar)

        # Write results to file
        with open(out_metrics_file, 'w') as metrics_file:  
            metric_writer = csv.writer(metrics_file)
            _ = metric_writer.writerow(['metric', 'value'])

            for input_key, key_val in metrics.items():
                _ = metric_writer.writerow([input_key, key_val])

        # If doing embedding visualization, save the figure to file
        # and save the embeddings (original, not tSNE) themselves as well
        # TODO: resize figure?
        if(args.do_embed_vis):
            embed_suffix = 'pathway' if args.use_pathway_embeds else 'prelinfinal'
            out_embed_file = os.path.join(metrics_dir, f'embeddings_{embed_suffix}.csv')

            embed_suffix += '_'+args.embed_vis_method

            if fig is not None:
                fig.savefig(os.path.join(metrics_dir, f'embeddings_{embed_suffix}.png'))
                fig.savefig(os.path.join(metrics_dir, f'embeddings_{embed_suffix}.svg'))

            print(f"Saving embeddings to {out_embed_file} (and figures with png/svg extensions)...\n")

            with open(out_embed_file, 'w') as embed_file:  
                metric_writer = csv.writer(embed_file)
                _ = metric_writer.writerow(['embeds', 'label'])

                for embed, label in zip(*embeds_and_labels):
                    _ = metric_writer.writerow([embed.tolist(), label])


    if not args.skip_explain:
        print(f"Basic explanation of model now...\n")
        vis_dir = os.path.join(args.out_dir, 'explanation_visualization')
        os.makedirs(vis_dir, exist_ok=True)

        # Need to create a dataloader for everyone if we are explaining on the entire dataset
        # If not explaining on everyone, then just use the test data
        if args.explain_all:
            explain_data_used = data
        else:
            explain_data_used = test_data

        explain_dl = torch_geometric.loader.DataLoader(explain_data_used, shuffle=False, batch_size=args.explain_batch_size, drop_last=False)

        input_related_counts, class_related_counts = \
            explanation_functions.alt_explain_on_dataset(model, explain_dl, full_graph_metadata, device, 
                                                     verbose=args.enable_progress_bar, 
                                                     topk_input=args.topk_input, topk_class=args.topk_class, 
                                                     vis_dir=vis_dir, do_vis=args.do_explain_vis,
                                                     explain_only_correct_pos=args.explain_only_correct_pos)
        
        
        total_pos_input_counts, total_neg_input_counts, total_abs_input_counts, total_all_input_counts = input_related_counts 
        total_top_class_counts, all_top_class_mags = class_related_counts

        num_examples = total_all_input_counts.get('Total_explained_examples', 1)

        print(f"\n~~~~~\n(All) contributing input node counts at {num_examples} explained examples:\n{total_abs_input_counts}\n~~~~~\n")
        print(f"\n~~~~~\nImportant class node counts at {num_examples} explained examples:\n{total_top_class_counts}\n~~~~~\n")

        # Write results to file
        pref = 'entiredata' if args.explain_all else 'testonly'
        pref += '_correctposonly' if args.explain_only_correct_pos else '_regardeverything'
        out_explain_input_file = os.path.join(vis_dir, f'{pref}_top_imp_connected_input_counts.csv')
        out_explain_class_file = os.path.join(vis_dir, f'{pref}_top{args.topk_class}_imp_class_counts_mag.csv')

        with open(out_explain_input_file, 'w') as exp_input_file:  
            exp_input_writer = csv.writer(exp_input_file)
            _ = exp_input_writer.writerow(['input_name', 'total_count', 'abs_important_count', 'pos_important_count', 'neg_important_count'])

            for input_key, all_count_val in total_all_input_counts.items():
                abs_imp_count_val = total_abs_input_counts.get(input_key, 0)
                pos_imp_count_val = total_pos_input_counts.get(input_key, 0)
                neg_imp_count_val = total_neg_input_counts.get(input_key, 0)

                _ = exp_input_writer.writerow([input_key, all_count_val, abs_imp_count_val, pos_imp_count_val, neg_imp_count_val])

        with open(out_explain_class_file, 'w') as exp_class_file:  
            exp_class_writer = csv.writer(exp_class_file)
            _ = exp_class_writer.writerow(['input_name', 'important_count', 'important_proportion', 'sum_magnitude'])

            for class_key, class_mag_val in all_top_class_mags.items():
                class_count_val = total_top_class_counts.get(class_key, 0)
                class_prop_val = class_count_val / num_examples

                _ = exp_class_writer.writerow([class_key, class_count_val, class_prop_val, class_mag_val])


    # Save the full dataset to the file now that all samples have been processed at least once
    # if a file was provided and we aren't skipping data saving
    if(reload_data_file is not None and not args.skip_data_save):
        print(f"Saving/dumping data now...\n")
        data._dump_data_to_file()

    
    ipdb.set_trace()



# Argument parsing for proper main call
if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Load the input data and train the GNN with it')


    # Input arguments

    parser.add_argument("--input-data-adj", type=str, 
            default='./data/adsp/graph_structure.pt',
            help="Input data graph structure file.")
    
    parser.add_argument("--input-data-features", type=str, 
            default='./data/adsp/collated_tensors_withPheno.pt',
            help="Input data node features file.")

    parser.add_argument("--train-prop", type=float,
            default=0.7, 
            help="Proportion of input data to use for training. "
            "Note that the validation proportion is determined by what remains after --train-prop and --test-prop.")

    parser.add_argument("--test-prop", type=float,
            default=0.15, 
            help="Proportion of input data to use for testing (checking after training completely finished). "
            "Note that the validation proportion is determined by what remains after --train-prop and --test-prop.")

    parser.add_argument("--force-cpu", action='store_true',
            help="Argument to force the use of a CPU instead of a GPU, even if a GPU is available. "
            "Usually done if getting OOM errors not fixed by changing the batch size or model architecture.")

    parser.add_argument("--push-raw-data", action='store_true',
            help="Argument to force the raw data to whichever device the model is stored on. "
            "For example, this will try to push all of the raw data to the GPU if the GPU is used.")

    parser.add_argument("--enable-multigpu", action='store_true',
            help="Argument that will enable the model to use multiple GPUs for processing. "
            "This is currently not supported because of issues with individual device-dispatches. "
            "This is a TODO feature, not to use.")
    
    parser.add_argument("--make-sparsed-nodes", action='store_true',
            help="Argument that will filter which nodes are present in the node feature data (>0 counts for variants), saving on computation.")

    parser.add_argument("--make-sparsed-edges", action='store_true',
            help="Argument that will make the graph data cast to sparse tensors, saving on computation.")

    parser.add_argument("--add-node-indegree-as-feat", action='store_true',
            help="Argument that will add node indegrees (divided by the max for a node type) as a feature for non-variant nodes. "
            "This will occur for EACH unique meta-triplet entering a node, and so may lead to differing feature numbers for each node. "
            "This is particularly useful if using --make-sparsed-nodes as it will encode the info for how many original variants existed.")

    # Model initialization parameters

    parser.add_argument("--model-type", type=str,
            default='hgt',
            help="Model type (options are 'hgt', 'han', and 'sage', default 'hgt').")
    
    # parser.add_argument("--no-compile", action='store_true',
    #         help="Argument to skip compiling the model (using torch_geomtric.compile). "
    #         "May be needed if compiling is causing major runtime issues or is crashing.")
    
    parser.add_argument("--basis-dim", type=int,
            default=None,
            help="Pass a value if would like input dimensions defined (if not, will assume 1). "
            "Note this will lead to there being additional linear layers included before the graph convolutions to get all nodes to the same number of features. "
            "Pass this if using --add-node-indegree-as-feat as that will result in different node feature numbers (one for each incoming edge type to each node, which varies from node type to node type). ")

    parser.add_argument("--hidden-dim", type=int,
            default=8,
            help="Dimension of hidden state features to be used during all convolutions.")
    
    parser.add_argument("--no-include-covs", action='store_true',
            help="Flag to NOT include individual covariates (sex, age, etc.).")
    
    parser.add_argument("--global-pool-type", type=str,
            default='none',
            help="Type of global pooling to include ('none' for no pooling by default). "
            "Options are 'mean', 'sum', and 'max' for pooling.")
    
    parser.add_argument("--dropout-prob", type=float,
            default=0.0,
            help="Dropout probability for the models (for regularization) - (default 0.0). "
            "Will perform dropout with the specified probability at each of the final classification linear layers. "
            "Note that with graph classification, the usefulness of this becomes more questionable (hence a default of 0.0).")

    parser.add_argument("--num-conv-layers", type=int,
            default=2,
            help="Number of stacked convolutional layers. Passed as num_conv_layers to the models.")

    parser.add_argument("--num-lin-layers", type=int,
            default=2,
            help="Number of stacked linear layers in the classification MLP. Passed as num_lin_layers to the models.")
    
    parser.add_argument("--lin-layer-feats", type=int,
            nargs='*',
            default=-1,
            help="Number of (output) features for each linear layer at the classification MLP. " 
            "Passing -1 will use the value of --hidden-dim until the final layer. "
            "A value should not be provided for the final layer as this will be based on the number of output classes.")

    parser.add_argument("--do-class-weighting", action='store_true',
            help="Pass if you want to do positive class reweighting based on the proportion in the dataset. "
            "Will scale the loss function by the inverse of [(class count) over (total number of classes times samples)].")

    # Training arguments (hyperparameters and such)

    parser.add_argument("--batch-size", type=int,
            default=4,
            help="Size of the batches to pass to the model and learn on at once.")

    parser.add_argument("--lr", type=float,
            default=1e-2,
            help="Learning rate of the model.")

    parser.add_argument("--epochs", type=int,
            default=500,
            help="Number of epochs for training.")
    
    parser.add_argument("--early-stop", type=int,
            default=10,
            help="Number of validation epochs without improvement before early stopping occurs. "
            "This functionally means --early-stop * --val-epochs number of training epochs before early stopping occurs.")

    parser.add_argument("--skip-training", action='store_true',
            help="Argument to skip training. "
            "Most useful with the --warm-start-model option to evaluate a previously trained model.")

    parser.add_argument("--warm-start-model", type=str,
            default=None,
            help="Argument to load a model that is passed as an argument for the training.")
    
    parser.add_argument("--optimizer", type=str,
            default='sgd',
            help="Optimizer (options are 'sgd' and 'adam', default 'sgd').")

    parser.add_argument("--data-reload-prefix", type=str,
            default=None,
            help="Argument to load existing data to prevent needing to reprocess OR " 
            "if the data does not exist, then will SAVE the data instead. "
            "Full file name is based on the prefix provided in the form [prefix].pt. "
            "Note that the seed is irrelevant to this as the data is stored as keys relative to INDICES in the original data.")

    parser.add_argument("--val-epochs", type=int,
            default=10,
            help="How often to do model validation - default every 100 epochs. "
            "Set to 0 to disable model validation during training.")


    # Output arguments

    parser.add_argument("--downstream-on-best", action='store_true',
            help="Argument is only used when a model is trained. "
            "This will explain and evaluate on the best model (based on validation loss) rather than the final model. ")


    parser.add_argument("--skip-eval", action='store_true',
            help="Argument to skip final evaluation on the validation dataset. "
            "Most useful with the --warm-start-model option to inspect a previously trained model.")

    parser.add_argument("--skip-data-save", action='store_true',
            help="Argument to skip saving the data at the end. "
            "Most useful with the --warm-start-model option to inspect a previously trained model.")

    parser.add_argument("--skip-explain", action='store_true',
            help="Argument to skip explaining the test data at the end. "
            "Most useful with the --warm-start-model option to inspect a previously trained model.")

    parser.add_argument("--out-dir", type=str,
            default='./gnn_output/',
            help="Output directory of model and other outputs."
            "Also where temporary files are stored.")

    parser.add_argument("--model_prefix", type=str, 
            default='trained_model',
            help="Prefix for the trained model file. "
            "Will be saved into the --out-dir as [prefix]_[epoch].pt.")

    # Evaluation arguments

    parser.add_argument("--do-embed-vis", action='store_true',
            help="Argument to do visualizations of the model's embeddings on the samples. "
            "Most useful with the --warm-start-model option to inspect a previously trained model."
            "Will only be used if --skip-eval is not passed.")
    
    parser.add_argument("--use-pathway-embeds", action='store_true',
            help="Argument to use the pathway embeddings for visualization instead of the final embeddings "
            "before the final linear classification layer. "
            "This will only be used if --skip-eval is not passed and --do-embed-vis is passed.")
    
    parser.add_argument("--embed-vis-method", type=str,
            default='tsne',
            help="Argument to use the pathway embeddings for visualization instead of the final embeddings "
            "before the final linear classification layer. "
            "This will only be used if --skip-eval is not passed and --do-embed-vis is passed.")

    # Explanation arguments
    parser.add_argument("--explain-batch-size", type=int,
            default=None,
            help="Batch size to use when explaining. "
            "Note that explaining requires a bit more memory than training, so this may need to be smaller than the training batch size. "
            "If not passed, then will use the training batch size --batch-size.")

    parser.add_argument("--topk-input", type=int,
            default=10,
            help="Number of top input nodes to consider for explanation. ")
    
    parser.add_argument("--topk-class", type=int,
            default=5,
            help="Number of top class nodes to consider for explanation. ")
    
    parser.add_argument("--do-explain-vis", action='store_true',
            help="Argument to do visualizations of all of the explanations. "
            "Most useful with the --warm-start-model option to inspect a previously trained model.")
    
    parser.add_argument("--explain-all", action='store_true',
            help="Argument to explain on the entire dataset, rather than just the holdout test set. "
            "This is arguably a better representation of importance, especially for rarer variants.")
    
    parser.add_argument("--explain-only-correct-pos", action='store_true',
            help="Argument to only explain the correctly, positively-classified examples. "
            "This is likely not that valuable overall, but is still an option. ")


    # Miscellaneous arguments (plotting, etc.)

    parser.add_argument("--enable-progress-bar", action='store_true', 
            help="Enable the TQDM progress bar. "
            "Note that TQDM does add some overhead, so disabling it is better on headless cluster systems. "
            "But this can be enabled for debugging/monitoring progress.")

    parser.add_argument("--seed", type=int,
            default=9,
            help="Random seed for training and dataset split.")





    # Parse arguments and handle conflicts/other issues with arguments
    args = parser.parse_args()

    # Define train/test/val proportions
    args.val_prop = 1 - (args.train_prop + args.test_prop)

    # Make output directory if does not exist
    os.makedirs(args.out_dir, exist_ok=True)

    # Warn the user if they used --add-node-indegrees-as-feat but not --basis_dim
    if(args.add_node_indegree_as_feat and not args.basis_dim):
        warnings.warn("Using --add-node-indegree-as-feat without defining a --basis-dim may lead to incompatible shape errors "
                      "in the model forward if the number of input edge types varies per node type. "
                      "Note that the 'sage' model may not have this issue, but defining a basis dimension is still better practice.")

    if(args.lin_layer_feats != -1 and len(args.lin_layer_feats)+1 != args.num_lin_layers):
        raise ValueError(f"Defined a different number of --lin-layer-feats ({args.lin_layer_feats}) as --num-lin-layers ({args.num_lin_layers})! " 
                         "Either pass -1 or feature numbers for each layer except the last one. ")


    # Set batch size for explanation if not passed
    if(args.explain_batch_size is None):
        args.explain_batch_size = args.batch_size


    print(args)
    print()
    print("Training AD genomics GNN model with the arguments above...")
    main(args)

