# gnn-genomics-AD
GNN-based analysis of genomics data for AD risk prediction (built on PyG)

This is the version as of submission of the code that was used to train  heterogeneous graph neural networks (GNNs) for Alzheimer's disease prediction and explanation of those GNNs thereof.

The main script to initialize a model and dataset, train the model, and explain on the model can be found in `main_gnn.py`. All of the modularized functions and classes for the GNN architectures, explaining code, and training/evaluation can be found in the folder `gnn_code`.

## Setting up

Please see the `requirements.txt` file to find dependencies. Note that a custom (publicly-available) implementation of Pytorch Geometric for heterogeneous explanations is required and can be found here: https://github.com/rachitk/pytorch_geometric/tree/het-captum-explain. This explanation code has been submitted as a pull request to Pytorch Geometric and is pending a minor rewrite before being merged into the main codebase.


## Running the code

Note that this repository does NOT include the processed ADSP data, as this dataset is only available by approval through the Data Sharing Service of the National Institute on Aging and we are unable to redistribute it. If the required data is placed into the `data` folder, then this script can be run directly.

All of the arguments to run `main_gnn.py` are documented close to the bottom of this script, on lines 370-600. An example command to train a GNN from scratch is:

```
python main_gnn.py --input-data-adj ./data/adsp/graph_structure.pt --input-data-features ./data/adsp/no_apoe_covars/collated_tensors_withPheno_noSepAPOE.pt --enable-progress-bar --batch-size 128 --hidden-dim 16 --make-sparsed-edges --make-sparsed-nodes --do-class-weighting --lr 1e-4 --add-node-indegree-as-feat --basis-dim 8 --val-epochs 5 --optimizer adam --num-lin-layers 6 --lin-layer-feats 256 128 64 32 16 --model-type han --data-reload-prefix ./data/adsp/processed_data_withindeg_withcovars_noAPOE --out ./gnn_out/gnn_out_sparsenode_hid16_basis8_lr1e-4_adamopt_lin256-128-64-32-16-out_nopool_han_noAPOE_v2_seed9 --seed 9 --skip-data-save --downstream-on-best --explain-all
```

An example command to warm-start from a previously-trained model and perform explanations is:

```
python main_gnn.py --input-data-adj ./data/adsp/graph_structure.pt --input-data-features ./data/adsp/no_apoe_covars/collated_tensors_withPheno_noSepAPOE.pt --enable-progress-bar --batch-size 128 --hidden-dim 16 --make-sparsed-edges --make-sparsed-nodes --do-class-weighting --lr 1e-4 --add-node-indegree-as-feat --basis-dim 8 --val-epochs 5 --optimizer adam --num-lin-layers 6 --lin-layer-feats 256 128 64 32 16 --model-type han --data-reload-prefix ./data/adsp/processed_data_withindeg_withcovars_noAPOE --out ./gnn_out/gnn_out_sparsenode_hid16_basis8_lr1e-4_adamopt_lin256-128-64-32-16-out_nopool_han_noAPOE_v2_seed9 --seed 9 --skip-training --warm-start-model ./gnn_out/gnn_out_sparsenode_hid16_basis8_lr1e-4_adamopt_lin256-128-64-32-16-out_nopool_han_noAPOE_v2_seed9/trained_model_best_val_loss.pt --skip-data-save --explain-all
```