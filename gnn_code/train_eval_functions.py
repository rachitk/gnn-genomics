import torch
import torch.nn as nn

import numpy as np

from tqdm import tqdm
from functools import partialmethod

import os

# Import from sklearn to calculate metrics
import sklearn
from sklearn import metrics

import ipdb



def model_evaluation(model, device, eval_data, loss_fn=None, coerce_binary=False, 
                     do_embed_vis=False, embed_vis_method='tsne', 
                     use_pathway_embeds=False, verbose=False):
    with torch.no_grad():
        # Set the model to evaluation mode
        model.eval()

        sigmoid_fn = nn.Sigmoid()

        eval_metric_dict = {}
        batch_metric_dicts_list = []
        total_loss = 0

        if(do_embed_vis):
            embeds = []
            labels = []

        pbar_eval = tqdm(eval_data, disable=not verbose, leave=False, unit='batch')
        pbar_eval.set_description("Evaluation")

        # Loop through the validation data
        for i, g_batch_eval in enumerate(pbar_eval):
            # Get the model output 
            if(do_embed_vis):
                g_batch_eval = g_batch_eval.to(device)
                if('adj_t' in g_batch_eval.keys()):
                    is_adj_t = True
                    eindex_dict = g_batch_eval.adj_t_dict
                else:
                    is_adj_t = False
                    eindex_dict = g_batch_eval.edge_index_dict
                
                class_batch_info = g_batch_eval[model.class_node].batch

                embed1 = model._gnn_model_forward(g_batch_eval.x_dict,
                                                 eindex_dict,
                                                 g_batch_eval.num_graphs,
                                                 is_adj_t,
                                                 class_batch_info
                                                 )
                
                embed2 = model._lin_model_to_preclass_layer_forward(embed1, g_batch_eval.covs)
                yhat = model._final_class_layer_forward(embed2)
                
                # embed1 are the raw GNN embeddings (at the pathway level)
                # as opposed to embed2, the embeddings just before the final classification
                if(use_pathway_embeds):
                    embeds.append(embed1)
                else:
                    embeds.append(embed2) 
                labels.append(g_batch_eval.y)
            else:
                yhat = model.forward_with_heterodata(g_batch_eval.to(device))

            y_eval = g_batch_eval.y

            # Calculate metrics from yhat (sigmoided) and y_eval (threshold 0.5 by default)
            batch_metric_dict = calc_metrics(y_eval.cpu(), sigmoid_fn(yhat).cpu())
            batch_metric_dicts_list.append(batch_metric_dict)

            # Compute loss and add to total loss
            if(loss_fn is not None):
                if(coerce_binary):
                    loss = loss_fn(yhat.squeeze(dim=1), y_eval.float())
                else:
                    loss = loss_fn(yhat, y_eval)

                total_loss += loss.item()

        if(loss_fn is not None and i > 0):
            eval_metric_dict['val_loss'] = total_loss / i

        keys = batch_metric_dicts_list[0].keys()

        for metric in keys:
            metric_vals = [d[metric] for d in batch_metric_dicts_list]
            eval_metric_dict[metric] = np.nanmean(metric_vals)

        # Set model back to training mode
        model.train()

        if(do_embed_vis):
            import matplotlib.pyplot as plt
            if embed_vis_method.lower() == 'tsne':
                # Use T-SNE to visualize embeddings
                print("Generating t-SNE visualization of individual embeddings...")
                from sklearn.manifold import TSNE

                tsne = TSNE(n_components=2, random_state=9, perplexity=10, verbose=2)
                embeds = torch.cat(embeds, dim=0).cpu().detach().numpy()
                labels = torch.cat(labels, dim=0).cpu().detach().numpy()
                embeds_tsne = tsne.fit_transform(embeds)
                fig, ax = plt.subplots(1, 1)

                # With seismic, blue is 0 and red is 1
                ax.scatter(embeds_tsne[:,0], embeds_tsne[:,1], c=labels, cmap='seismic', edgecolors=None, 
                           alpha=0.3, s=15)
                # ax.legend()
                ax.grid(True)

                ax.set_title("t-SNE Visualization of Individual Embeddings")
                ax.set_xlabel("t-SNE Component 1")
                ax.set_ylabel("t-SNE Component 2")
            
            elif embed_vis_method.lower() == 'pca':
                # Use PCA to visualize embeddings
                print("Generating PCA visualization of individual embeddings...")
                from sklearn.decomposition import PCA

                pca = PCA(n_components=2, random_state=9)
                embeds = torch.cat(embeds, dim=0).cpu().detach().numpy()
                labels = torch.cat(labels, dim=0).cpu().detach().numpy()
                embeds_pca = pca.fit_transform(embeds)
                fig, ax = plt.subplots(1, 1)

                # With seismic, blue is 0 and red is 1
                ax.scatter(embeds_pca[:,0], embeds_pca[:,1], c=labels, cmap='seismic', edgecolors=None, alpha=0.2)
                # ax.legend()
                ax.grid(True)

                ax.set_title("PCA Visualization of Individual Embeddings")
                ax.set_xlabel("PC1")
                ax.set_ylabel("PC2")

            else:
                print("Invalid embedding visualization method specified (tsne and pca supported). Returning embeds+labels without visualization.")
                return eval_metric_dict, None, (embeds, labels)

            return eval_metric_dict, fig, (embeds, labels)
            
        else:
            return eval_metric_dict, None, None



def calc_metrics(ytrue, yhat):
    # Note that yhat, as passed in, is PROBABILITY for each class (and is mutually exclusive)

    # Metrics dictionary that will be returned
    metric_dict = {}

    # Effectively assumes a critical threshold of 0.5 - can be adjusted if needed
    # Will handle multiclass vs. binary targets in this conditional
    if(yhat.shape[1] > 1):
        yhat_labels = torch.argmax(yhat, 1)
        yhat_probs = yhat[:,1]
    else:
        yhat_labels = (yhat.squeeze() > 0.5).long()
        yhat_probs = yhat

    # Uses probability predictions
    metric_dict['auroc'] = sklearn.metrics.roc_auc_score(ytrue, yhat_probs)
    metric_dict['log_loss'] = sklearn.metrics.log_loss(ytrue, yhat_probs)
    metric_dict['auprc'] = sklearn.metrics.average_precision_score(ytrue, yhat_probs)

    # Uses label predictions (thresholded from probabilities)
    metric_dict['precision'], metric_dict['recall'], metric_dict['fbeta'], _ = \
            sklearn.metrics.precision_recall_fscore_support(ytrue, yhat_labels, average='weighted', zero_division=0)
    # metric_dict['hamming'] = sklearn.metrics.hamming_loss(ytrue, yhat_labels)
    # metric_dict['zero_one_loss'] = sklearn.metrics.zero_one_loss(ytrue, yhat_labels)
    # metric_dict['jaccard_score'] = sklearn.metrics.jaccard_score(ytrue, yhat_labels, average='weighted', zero_division=0)
    metric_dict['balacc'] = sklearn.metrics.balanced_accuracy_score(ytrue, yhat_labels)


    return metric_dict



def training_loop(model, device, lr, epochs, train_data, val_data=None, label_weights=None, verbose=False, 
            save_epochs=50, save_location=None, save_prefix=None, val_epochs=0, coerce_binary=False, opt_type='sgd',
            early_stop_thresh=10):

    # Move model to device (data moved later)
    model = model.to(device)


    # Disable TQDM progress bar if user does not explicitly request it
    tqdm.__init__ = partialmethod(tqdm.__init__, disable=not verbose)

    # Define loss function and optimizer (may make these selectable in the future?)
    if(coerce_binary):
        # Binary loss function
        loss_func = nn.BCEWithLogitsLoss(pos_weight=label_weights.to(device)).to(device)
    else:
        # Multiclass loss function
        loss_func = nn.CrossEntropyLoss(weight=label_weights.to(device)).to(device)


    # Optimizer (SGD/Adam)
    if(opt_type.lower() == 'sgd'):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    elif(opt_type.lower() == 'adam'):
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # Define whether validation dataloader exists
    val_dl_exist = val_data is not None
    # Set val_epochs to either val_epochs if it's not 0 (or None), if it is then set it to the number of epochs+1 (so no val)
    val_epochs = val_epochs if val_epochs else epochs+1
    best_val_loss = torch.inf
    num_epochs_no_improvement = 0

    # Actual training loop

    # Set up TQDM progress bar for the entire training loop
    pbar_train = tqdm(range(epochs), unit="epoch")
    pbar_train.set_description("Training")

    for epoch in pbar_train:
        model.train()

        # List to store losses for this epoch, used below for mean loss over epoch
        losses = []

        # Need a more concrete check here since it could just be that the dataloader is empty
        # Also need to consider how often to evaluate (every epoch? every 10 epochs?)
        # Currently set to every 10 epochs
        do_val = val_dl_exist and (((epoch+1) % val_epochs) == 0)

        # Batch processing
        # Can reset this value based on a certain number of epochs (every 10th epoch, keep the pbar or such)
        keep_pbar = True

        # Set up TQDM progress bar for this epoch 
        # (not the iterable directly as we want to update the bar after we're finished iterating)
        with tqdm(range(len(train_data)), leave=keep_pbar) as pbar_epoch:
            pbar_epoch.set_description("Epoch {:05d}".format(epoch))

            for i, g_batch_train in enumerate(train_data):
                # Move data to device
                g_batch_train = g_batch_train.to(device)

                # Reset gradients (for batch-based optimization)
                optimizer.zero_grad()

                # Get the model's outputs and calculate the loss
                yhat = model.forward_with_heterodata(g_batch_train)

                if(coerce_binary):
                    loss = loss_func(yhat.squeeze(dim=1), g_batch_train.y.float())
                else:
                    loss = loss_func(yhat, g_batch_train.y)

                # Send the loss backwards to optimize the parameters
                # TODO: consider doing this after some number of batches
                loss.backward()
                optimizer.step()

                # Store losses for mean epoch loss and report in the tqdm progressbar
                losses.append(loss.item())
                pbar_epoch.set_postfix(Loss='{:.4f}'.format(loss.item()))
                pbar_epoch.update(1)

            # Report the average loss for this epoch once the epoch is done
            pbar_epoch.set_postfix(train_loss='{:.4f}'.format(np.mean(losses)))
            pbar_epoch.refresh()



            # If validation data was passed, do validation here (evaluate on validation data)
            if(do_val):
                metric_dict, _, _ = model_evaluation(model, device, val_data, loss_fn=loss_func, 
                                               coerce_binary=coerce_binary, verbose=verbose)

                pbar_epoch.set_postfix(train_loss='{:.4f}'.format(np.mean(losses)), **metric_dict)
                pbar_epoch.refresh()

                # Make decision based on metrics, if any required
                if(metric_dict['val_loss'] < best_val_loss):
                    best_val_loss = metric_dict['val_loss']
                    torch.save(model.state_dict(), os.path.join(save_location, '{}_best_val_loss.pt'.format(save_prefix)))
                    num_epochs_no_improvement = 0
                else:
                    num_epochs_no_improvement += 1

            if(save_epochs and save_location):
                if(((epoch+1) % save_epochs) == 0):
                    torch.save(model.state_dict(), os.path.join(save_location, '{}_{}.pt'.format(save_prefix, str(epoch))))

            # Early stopping condition
            if(num_epochs_no_improvement >= early_stop_thresh):
                optimizer.zero_grad()
                pbar_train.close()
                print("Early stopping reached at epoch {} after {} validation epochs with no improvement!".format(epoch, early_stop_thresh))
                break

        # TODO: Update progress bar for the training loop (if any needed)

    # Return the fully trained model after zeroing gradients again
    optimizer.zero_grad()
    
    return model




