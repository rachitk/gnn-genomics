# gnn-genomics-AD
GNN-based analysis of genomics data for AD risk prediction (built on PyG)

This is the version as of submission of the code that was used to train  heterogeneous graph neural networks (GNNs) for Alzheimer's disease prediction and explanation of those GNNs thereof.

The main script to initialize a model and dataset, train the model, and explain on the model can be found in `main_gnn.py`. All of the modularized functions and classes for the GNN architectures, explaining code, and training/evaluation can be found in the folder `gnn_code`.

## Setting up

You should first clone the repository and enter it:

```
git clone https://github.com/rachitk/gnn-genomics
cd gnn-genomics
```

To install dependencies, make sure you are using a Python version between 3.8 and 3.11; you can then run the following command (we recommend you run this in a virtual environment or Conda environment):

```
pip install -r requirements.txt
```

Please see the `requirements.txt` file to find a list of dependencies.

Note that a custom (publicly-available) implementation of Pytorch Geometric for heterogeneous explanations is required and can be found here: [https://github.com/rachitk/pytorch_geometric/tree/het-captum-explain-paper](https://github.com/rachitk/pytorch_geometric/tree/het-captum-explain-paper). This explanation code has been submitted as a pull request to Pytorch Geometric and is pending a minor rewrite before being merged into the main codebase (the PR version is at a different branch, [https://github.com/rachitk/pytorch_geometric/tree/het-captum-explain](https://github.com/rachitk/pytorch_geometric/tree/het-captum-explain), though to reproduce the paper, you should use the above version).

Note that most of the versions are pinned in part due to the dependency on the above custom version of Pytorch Geometric, though newer versions of several dependencies should work once the Pytorch Geometric pull request is accepted (as a newer version of Pytorch and Pytorch Geometric can be used after that).


## Running the code

### Toy dataset

For the purposes of review/verification of the code, we provide a toy dataset (and the code used to make this dataset) that can be used in the `example_data` folder instead. A command that uses these files can be found below:

```
python main_gnn.py --input-data-adj ./example_data/example_graph_structure.pt --input-data-features ./example_data/example_collated_tensors_withPheno_noSepAPOE.pt --enable-progress-bar --batch-size 128 --hidden-dim 16 --make-sparsed-edges --make-sparsed-nodes --do-class-weighting --lr 1e-4 --add-node-indegree-as-feat --basis-dim 8 --val-epochs 5 --optimizer adam --num-lin-layers 6 --lin-layer-feats 256 128 64 32 16 --model-type han --data-reload-prefix ./example_data/example_processed_data_withindeg_withcovars_noAPOE --out ./gnn_out/example_gnn_out_sparsenode_hid16_basis8_lr1e-4_adamopt_lin256-128-64-32-16-out_nopool_han_noAPOE_v2_seed9 --seed 9 --skip-data-save --downstream-on-best --explain-all --do-embed-vis --embed-vis-method pca --epochs 20
```

Note that this toy dataset won't really produce any meaningful outputs due to the definition of the phenotype in it being extraordinarily simple - it just provides an example of what the expected input is and allows one to run the code end-to-end to verify that it runs.


### ADSP Dataset

Note that this repository does NOT include the processed ADSP data, as this dataset is only available by approval through the Data Sharing Service of the National Institute on Aging and we are unable to redistribute it. If the required data is placed into a `data/adsp` folder, then this script can be run directly. We are happy to provide the code that we used to process the ADSP dataset into the form mocked-up by the toy dataset above if one can provide their ADSP project approval (as the conversion code does include metadata that we are not allowed to redistribute).

All of the arguments to run `main_gnn.py` are documented close to the bottom of this script, on lines 370-600. An example command to train a GNN from scratch is:

```
python main_gnn.py --input-data-adj ./data/adsp/graph_structure.pt --input-data-features ./data/adsp/no_apoe_covars/collated_tensors_withPheno_noSepAPOE.pt --enable-progress-bar --batch-size 128 --hidden-dim 16 --make-sparsed-edges --make-sparsed-nodes --do-class-weighting --lr 1e-4 --add-node-indegree-as-feat --basis-dim 8 --val-epochs 5 --optimizer adam --num-lin-layers 6 --lin-layer-feats 256 128 64 32 16 --model-type han --data-reload-prefix ./data/adsp/processed_data_withindeg_withcovars_noAPOE --out ./gnn_out/gnn_out_sparsenode_hid16_basis8_lr1e-4_adamopt_lin256-128-64-32-16-out_nopool_han_noAPOE_v2_seed9 --seed 9 --skip-data-save --downstream-on-best --explain-all
```

An example command to warm-start from a previously-trained model and perform explanations is:

```
python main_gnn.py --input-data-adj ./data/adsp/graph_structure.pt --input-data-features ./data/adsp/no_apoe_covars/collated_tensors_withPheno_noSepAPOE.pt --enable-progress-bar --batch-size 128 --hidden-dim 16 --make-sparsed-edges --make-sparsed-nodes --do-class-weighting --lr 1e-4 --add-node-indegree-as-feat --basis-dim 8 --val-epochs 5 --optimizer adam --num-lin-layers 6 --lin-layer-feats 256 128 64 32 16 --model-type han --data-reload-prefix ./data/adsp/processed_data_withindeg_withcovars_noAPOE --out ./gnn_out/gnn_out_sparsenode_hid16_basis8_lr1e-4_adamopt_lin256-128-64-32-16-out_nopool_han_noAPOE_v2_seed9 --seed 9 --skip-training --warm-start-model ./gnn_out/gnn_out_sparsenode_hid16_basis8_lr1e-4_adamopt_lin256-128-64-32-16-out_nopool_han_noAPOE_v2_seed9/trained_model_best_val_loss.pt --skip-data-save --explain-all
```
