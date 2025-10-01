import torch
from torch import nn
import matplotlib.pyplot as plt
import seaborn as sns
import wandb
import time # for timestamp in wandbf
from torchmetrics import ConfusionMatrix
from torch.optim.lr_scheduler import StepLR
import sys
import os.path as osp
import os
import glob
import pickle
from sklearn.model_selection import KFold
from torch.utils.data import DataLoader, TensorDataset
import random
from preprocess2_og import get_dataloaders2, get_dataloaders2_for_cross_val
from Transformer_og import TransformerEncoder

# Global Hyperparameters
"""
#Normal train (no cross val) hyperparameters:
directory = r"D:\Final Project\TASK_RH_vis2\dataset"
task = 'rest'
batch_size = 16
epochs =  55
num_heads = 4
dropout =  0.7
weight_decay = 0.01
embedding_dim = 512
learning_rate = 0.000001
NET_list = ['Vis'] #,'Default_pCunPCC','Default_PFC','Default_Temp'] # Tali - 9_7 ['Default_Avg','Vis','Default_PFC','DorsAttn_Post', 'Default_pCunPCC', 'Default_Temp']
NET_indexes = [1] #[1,2,3,4,5,6,7]
H_list = ['BOTH'] #['RH', 'LH','BOTH']  # options: ['RH', 'LH']
Avg = 2
Create_synthetic_subjects = 1
n_synthetic_subjects = 10
use_original_for_val_test = True
Group_by_subjects = 0
group_size = 4
slice = 'end'
noise_level = 0.01
n_synthetic_TRs = 0
k = 4
run_mode = 'train'  # 'train' or 'cross_val'
timestamp = time.strftime("%d%m-%H%M")  # timestamp for wandb

"""
# cross_val hyperparameters
directory = r"D:\Final Project\TASK_RH_vis2\dataset"
task = 'rest'
batch_size = 8
epochs = 10
num_heads = 2
dropout = 0.7
weight_decay = 0.01
embedding_dim = 512
learning_rate = 0.00005
NET_list = ['Vis'] #,'Default_pCunPCC','Default_PFC','Default_Temp'] # Tali - 9_7 ['Default_Avg','Vis','Default_PFC','DorsAttn_Post', 'Default_pCunPCC', 'Default_Temp']
NET_indexes = [1] #[1,2,3,4,5,6,7]
H_list = ['BOTH'] #['RH', 'LH','BOTH']  # options: ['RH', 'LH']
Avg = 2
Create_synthetic_subjects = 0
n_synthetic_subjects = 10
use_original_for_val_test = True
Group_by_subjects = 1
group_size = 4
slice = 'end'
noise_level = 0.01
n_synthetic_TRs = 0
k = 4
run_mode = 'cross_val'  # 'train' or 'cross_val'
timestamp = time.strftime("%d%m-%H%M")  # timestamp for wandb


# List of input files to run on
exists_list = ['Default_1_BOTH',
'Default_pCunPCC_1_BOTH',
'Default_PFC_1_BOTH',
'Default_Temp_1_BOTH',
'DorsAttn_Post_1_BOTH',
'Vis_1_BOTH',
'Vis_2_RH',
'Vis_2_LH',
'Vis_3_RH',
'Vis_3_LH',
'Vis_4_RH',
'Vis_4_LH',
'Vis_5_RH',
'Vis_5_LH',
'Vis_6_RH',
'Vis_6_LH',
'Default_PFC_1_LH',
'Default_PFC_2_LH',
'Default_PFC_3_LH',
'DorsAttn_Post_4_RH',
'DorsAttn_Post_4_LH',
'DorsAttn_Post_5_RH',
'DorsAttn_Post_5_LH',
'DorsAttn_Post_6_RH',
'DorsAttn_Post_6_LH',
'Default_pCunPCC_1_RH',
'Default_pCunPCC_1_LH',
'Default_pCunPCC_2_RH',
'Default_pCunPCC_2_LH',
'Default_pCunPCC_3_RH',
'Default_pCunPCC_3_LH',
'Default_pCunPCC_4_RH',
'Default_pCunPCC_4_LH',
'Default_pCunPCC_5_RH',
'Default_pCunPCC_5_LH',
'Default_pCunPCC_6_RH',
'Default_pCunPCC_6_LH',
'Default_Temp_5_RH',
'Default_Temp_5_LH',
'Default_Temp_6_RH',
'Default_Temp_6_LH',
'Default_Temp_7_LH']

# Calculate metrics such as precision, recall, and f1-score
def calc_metrics(predicted_labels, true_labels, flag=False):
    """    Calculate performance metrics including precision, recall, and f1-score for each class,
    as well as the overall accuracy."""
    class_labels = set(true_labels)
    metrics = {}
    for label in class_labels:
        tp = sum((p == label and t == label) for p, t in zip(predicted_labels, true_labels))
        fp = sum((p == label and t != label) for p, t in zip(predicted_labels, true_labels))
        fn = sum((p != label and t == label) for p, t in zip(predicted_labels, true_labels))

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        metrics[label] = {
            'precision': precision,
            'recall': recall,
            'f1-score': f1
        }

    correct_predictions = sum(p == t for p, t in zip(predicted_labels, true_labels))
    accuracy = correct_predictions / len(predicted_labels)
    if flag:
        plt.figure(figsize=(10, 7))
        target = torch.tensor(true_labels)
        preds = torch.tensor(predicted_labels)
        confmat = ConfusionMatrix(task="multiclass", num_classes=14)
        cm = confmat(preds, target)
        cm_heatmap = sns.heatmap(cm, cbar=True, annot=True)
        figure_cm = cm_heatmap.get_figure()
        wandb_cm = wandb.Image(figure_cm, caption="Confusion Matrix | Movies")
        return {'Accuracy': accuracy, 'Confusion Matrix': wandb_cm}
    else:
        return {'Accuracy': accuracy}

# Main training loop
def train_loop(train_dataloader, eval_dataloader, test_dataloader, num_voxels, model_path, run_mode):
    """
    The main training loop for training and evaluating the Transformer model.
    Returns:
    tuple: A tuple containing the test accuracy, train loss, and train accuracy.
    """
    global num_heads, learning_rate, epochs, batch_size, dropout, weight_decay, embedding_dim
    model = nn.Sequential(
        TransformerEncoder(num_voxels=num_voxels, classes=14, time2vec_dim=1, num_heads=num_heads,
                           head_size=embedding_dim, ff_dim=embedding_dim,
                           num_layers=2,
                           dropout=dropout))
    model = model.float()
    device = torch.device('cuda:0')
    model.to(device)
    print('cuda: ', torch.cuda.is_available())

    best_loss = 100
    best_train_loss = 100  # To store the best train loss in cross_val mode
    # Defining loss function and optimizer
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.9) #for learning rate decay

    wandb.watch(model)

    model.train()

    for epoch in range(epochs):
        lr_scheduler.step()
        train_losses = []
        train_pds, train_gts = [], []
        optimizer.zero_grad()

        with torch.set_grad_enabled(True):
            for idx, train_batch in enumerate(train_dataloader):
                data = train_batch[0]
                data = data.double()
                data = data.cuda()
                gt = train_batch[1].cuda()
                outputs = model(data.float())

                train_gts.extend(torch.argmax(gt, 1).tolist())
                train_pds.extend(torch.argmax(outputs, 1).tolist())
                loss = loss_fn(outputs, gt)
                train_losses.append(loss.item())
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                if idx % 100 == 0:
                    wandb.log({'Train/loss': loss.item(),
                               'Train/epoch': epoch,
                               'Train/step': idx})
            if run_mode == 'cross_val' and sum(train_losses) / len(train_losses) < best_train_loss:
                torch.save({"model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict()
                            }, model_path)
                best_train_loss = sum(train_losses) / len(train_losses)
        # Printing the training loss
        print(
            f"Epoch [{epoch + 1}/{epochs}], Loss: {sum(train_losses) / len(train_losses)}, Learning Rate: {optimizer.param_groups[0]['lr']}")
        wandb.log({'Train/loss per epoch': sum(train_losses) / len(train_losses),
                   'Train/epoch': epoch,
                   'Train/step': idx})

        for metric in calc_metrics(train_pds, train_gts).items():
            wandb.log({f'Train/{metric[0]}': metric[1]})

        if run_mode == 'train' and eval_dataloader: # No validation needed in cross validation (only training and test)
            print('Validation')
            eval_losses = []
            eval_pds, eval_gts = [], []
            for idx_val, eval_batch in enumerate(eval_dataloader):
                model.eval()
                eval_data = eval_batch[0]
                eval_data = eval_data.double()
                eval_data = eval_data.cuda()
                eval_gt = eval_batch[1].cuda()
                eval_output = model(eval_data.float())

                eval_gts.extend(torch.argmax(eval_gt, 1).tolist())
                eval_pds.extend(torch.argmax(eval_output, 1).tolist())
                eval_loss = loss_fn(eval_output, eval_gt)
                eval_losses.append(eval_loss.item())

                if idx_val % 100 == 0:
                    wandb.log({'Eval/loss': eval_loss,
                               'Eval/step': idx_val})

            if sum(eval_losses) / len(eval_losses) < best_loss:
                torch.save({"model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict()
                            }, model_path)
                best_loss = sum(eval_losses) / len(eval_losses)
                print('saving best model')
            print(f"Eval Loss: {best_loss}")
            # calculating accuracy
            for metric in calc_metrics(eval_pds, eval_gts).items():
                wandb.log({f'Eval/{metric[0]}': metric[1]})


    test_losses = []
    test_pds, test_gts = [], []
    print('Testing')
    test_model = model
    checkpoint = torch.load(model_path)
    test_model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    test_model.to(device)
    for idx_test, test_batch in enumerate(test_dataloader):
        test_model.eval()
        test_data = test_batch[0]
        test_data = test_data.double()
        test_data = test_data.cuda()
        test_gt = test_batch[1].cuda()
        test_output = test_model(test_data.float())

        test_gts.extend(torch.argmax(test_gt, 1).tolist())
        test_pds.extend(torch.argmax(test_output, 1).tolist())
        test_loss = loss_fn(test_output, test_gt)
        test_losses.append(test_loss.item())

    print(f"Test Loss: {sum(test_losses) / len(test_losses)}")
    wandb.log({'Test/loss': sum(test_losses) / len(test_losses)})
    for metric in calc_metrics(test_pds, test_gts, True).items():
        wandb.log({f'Test/{metric[0]}': metric[1]})

    # logging results to wandb
    if idx_test % 100 == 0:
        wandb.log({'Test/loss': test_loss,
                   'Test/step': idx_test})

    return calc_metrics(test_pds, test_gts)['Accuracy'], sum(train_losses) / len(train_losses), calc_metrics(train_pds, train_gts)['Accuracy']

# Cross-validation function
def run_cross_validation():
    """
    Run k-fold cross-validation on the dataset and evaluate model performance.
    Returns:
    list: List of test accuracies for each fold.
    """
    phase_path = osp.join(directory, 'cross_val', task)
    subject_folders = sorted(glob.glob(phase_path + '/**'))

    # Remove existing SYNTH_ and GROUP_ directories before running cross-validation
    for folder in subject_folders:
        if 'SYNTH_' in folder or 'GROUP_' in folder:
            print(f"Removing synthetic/group subject folder: {folder}")
            for file in glob.glob(osp.join(folder, '*')):
                os.remove(file)
            os.rmdir(folder)
    subject_folders = [folder for folder in subject_folders if 'SYNTH_' not in folder and 'GROUP_' not in folder]

    data_files = [osp.join(folder, f'{H}_{NET}_Avg.pkl') for folder in subject_folders]

    kf = KFold(n_splits=k)
    fold_accuracies = []
    train_accuracies = []

    wandb.login()
    timestamp = time.strftime("%d%m-%H%M")
    wandb.init(
        project="fmri_project",
        group='encoder_nets',
        name=f'cross_val_{timestamp}',
        config={
            "learning_rate": learning_rate, "epochs": epochs, "batch_size": batch_size, "dropout": dropout,
            "loss": 'CE', "optimizer": 'Adam',
            'attention heads': num_heads,
            "embedding dim": embedding_dim
        }
    )

    for fold, (train_index, test_index) in enumerate(kf.split(data_files)):
        run_suffix = f'run_{fold + 1}'
        train_files = [data_files[i] for i in train_index]
        test_files = [data_files[i] for i in test_index]

        train_dataloader, num_voxels = get_dataloaders2_for_cross_val(train_files, NET, NET_idx, H, batch_size, slice, task, Avg,
                                                                   noise_level, n_synthetic_TRs, Group_by_subjects, group_size, run_suffix)
        test_dataloader, _ = get_dataloaders2_for_cross_val(test_files, NET, NET_idx, H, batch_size, slice, task, Avg,
                                                         noise_level, n_synthetic_TRs, Group_by_subjects, group_size, run_suffix)

        if train_dataloader is None or test_dataloader is None:
            print(f"Skipping fold {fold + 1} due to missing data.")
            continue

        model_path = f'models/best_model_fold_{fold + 1}.pth'
        fold_accuracy, train_loss, train_accuracy = train_loop(train_dataloader, None, test_dataloader, num_voxels, model_path, 'cross_val')

        fold_accuracies.append(fold_accuracy)
        train_accuracies.append(train_accuracy)
        print(f"Fold {fold + 1} - Train Accuracy: {train_accuracy}, Test Accuracy: {fold_accuracy}")

    average_accuracy = sum(fold_accuracies) / len(fold_accuracies)
    print(f'Average k-fold accuracy: {average_accuracy}')
    wandb.log({'Average k-fold accuracy': average_accuracy})
    average_train_accuracy = sum(train_accuracies) / len(train_accuracies)
    print(f'Average train accuracy: {average_train_accuracy}')
    wandb.log({'Average train accuracy': average_train_accuracy})
    wandb.finish()

# Training function for normal train mode
def train():
    """
    Train the Transformer model on the dataset with the given configuration.

    - Logs the training process to WandB.
    - Saves the best model based on evaluation accuracy.

    Returns:
    tuple: Tuple containing the test accuracy, train loss, and train accuracy.
    """
    global num_heads
    inputs = []
    outputs = []

    train_dataloader, eval_dataloader, test_dataloader, num_voxels = get_dataloaders2(directory, NET, NET_idx, H,
                                                                                      slice, batch_size, task,
                                                                                      Avg=Avg, noise_level=noise_level, n_synthetic_TRs=n_synthetic_TRs,
                                                                                      Create_synthetic_subjects=Create_synthetic_subjects, n_synthetic_subjects=n_synthetic_subjects,
                                                                                      Group_by_subjects=Group_by_subjects, group_size=group_size,
                                                                                      use_original_for_val_test=use_original_for_val_test)

    if train_dataloader is None or eval_dataloader is None or test_dataloader is None:
        print(f"Skipping {H}_{NET}_{NET_idx} as files are missing.")
        sys.exit()

    wandb.login()
    timestamp = time.strftime("%d%m-%H%M")
    wandb.init(
        project="fmri_project",
        group='encoder_nets',
        name=f'{NET}_{NET_idx}_{H}_{timestamp}_avg_{Avg}',
        config={
            "learning_rate": learning_rate, "epochs": epochs, "batch_size": batch_size, "dropout": dropout,
            "loss": 'CE', "optimizer": 'Adam',
            'attention heads': num_heads,
            "embedding dim": embedding_dim
        }
    )

    model_path = f'models/best_model_{NET}_{NET_idx}_{H}.pth'
    train_loop(train_dataloader, eval_dataloader, test_dataloader, num_voxels, model_path, run_mode)
    wandb.finish()

# Loop for running normal training or cross-validation based on configuration
for NET in NET_list:
    for NET_idx in NET_indexes:
        for H in H_list:
            if f'{NET}_{NET_idx}_{H}' not in exists_list:
                continue
            print(
            f"Running training on {H}_{NET}_{NET_idx} for {epochs} epochs - Batch Size: {batch_size}, Learning Rate: {learning_rate}")
            if run_mode == 'cross_val':
                run_cross_validation()
            else:
                train()