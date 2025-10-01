import torch
from torch import nn
import matplotlib.pyplot as plt
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
from preprocess2 import get_dataloaders2, get_dataloaders2_for_cross_val
from preprocess2 import get_dataloaders2_random_split
from Transformer import TransformerEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd




import torch.nn.functional as F
os.makedirs("models", exist_ok = True)
# Global Hyperparameters
"""
#Normal train (no cross val) hyperparameters:
directory = r"D:\Final Project\TASK_RH_vis2\dataset"
task = 'movies'
batch_size = 16
epochs =  30
num_heads = 4
dropout =  0.5
weight_decay = 0
embedding_dim = 512
learning_rate = 0.00001
NET_list = ['Vis','Default_pCunPCC'] #,'Default_pCunPCC','Default_PFC','Default_Temp'] # Tali - 9_7 ['Default_Avg','Vis','Default_PFC','DorsAttn_Post', 'Default_pCunPCC', 'Default_Temp']
NET_indexes = [5] #[1,2,3,4,5,6,7]
H_list = ['RH'] #['RH', 'LH','BOTH']  # options: ['RH', 'LH']
Avg = 0
Create_synthetic_subjects = 0
n_synthetic_subjects = 10
use_original_for_val_test = True
Group_by_subjects = 0
group_size = 4
slice = 'start'
noise_level = 0.01
n_synthetic_TRs = 0
k = 4
run_mode = 'cross_val'  # 'train' or 'cross_val'
timestamp = time.strftime("%d%m-%H%M")  # timestamp for wandb
"""
#"""
# cross_val hyperparameters
#directory = r"D:\Final Project\TASK_RH_vis2\dataset"
directory = r"F:\HCP_DATA"
# Load the data
df = pd.read_csv("all_data.csv")
# Filter the data where '7T_RS-fMRI_Count' is 4
df_7T = df.loc[df['7T_RS-fMRI_Count'] == 4]
# Reset the index and assign it back to df_7T
df_7T = df_7T.reset_index(drop=True)

#task = 'rest'
batch_size = 8
epochs = 500
num_heads = 4
dropout = 0.1
weight_decay = 0
embedding_dim = 256
learning_rate = 1e-3
test=input("choose from 1:fluid intelligence, 2:memory, 3:personality, all:all is all man: ")
if test == "1":
    NET_list = ['Default_PFC']
    NET_indexes = [2]
elif test == "2":
    pass
    NET_list = ['Default']
elif test == "3":
    NET_list = ['Default']
    #NET_indexes = [5]
elif test == "all":
    pass
else:
    print("c'mon man.... play along")
#NET_list = ['Default']
#,'Default_pCunPCC','Default_PFC','Default_Temp'] # Tali - 9_7 ['Default_Avg','Vis','Default_PFC','DorsAttn_Post', 'Default_pCunPCC', 'Default_Temp']
#NET_indexes = [1] #[1,2,3,4,5,6,7]
H_list = ['LH'] #['RH', 'LH','BOTH']  # options: ['RH', 'LH']
Avg = 1
Create_synthetic_subjects = 0
n_synthetic_subjects = 0
use_original_for_val_test = True
Group_by_subjects = 0
group_size = 4
slice = 'end'
noise_level = 0.01
n_synthetic_TRs = 0
k = 4
run_mode = 'train'  # 'train' or 'cross_val'
timestamp = time.strftime("%d%m-%H%M")  # timestamp for wandb
#"""

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
'Default_PFC_4_LH',
'Default_PFC_5_LH',
'Default_PFC_6_LH',
'Default_PFC_7_LH',
'Default_PFC_8_LH',
'Default_PFC_9_LH',
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

def safe_load_model(model_path, model, num_voxels):
    if not os.path.exists(model_path):
        return model

    checkpoint = torch.load(model_path)
    try:
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print("Loaded existing model successfully.")
        return model
    except RuntimeError as e:
        if "size mismatch" in str(e):
            print("Model architecture mismatch. Deleting old model and starting fresh.")
            os.remove(model_path)
        else:
            raise e
    return model

def calc_metrics_per_output(predicted, actual):
    predicted = np.array(predicted)
    actual = np.array(actual)

    n_outputs = predicted.shape[1]
    results = []

    for i in range(n_outputs):
        y_true = actual[:, i]
        y_pred = predicted[:, i]
        mae = mean_absolute_error(y_true, y_pred)
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_true, y_pred)
        print(f"i= {i} y_true= {y_true} y_pred= {y_pred} \n")
        results.append({
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        })
    return results


# Calculate metrics such as precision, recall, and f1-score
def calc_metrics(predicted_labels, true_labels, flag=False):
    """ מחשב מדדי רגרסיה במקום מדדי סיווג """
    predicted_labels = np.array(predicted_labels).flatten()
    true_labels = np.array(true_labels).flatten()

    mae = mean_absolute_error(true_labels, predicted_labels)
    mse = mean_squared_error(true_labels, predicted_labels)
    rmse = np.sqrt(mse)
    r2 = r2_score(true_labels, predicted_labels)
    #for i in range(0,len(predicted_labels)):
    #    print(f"subject {i}: real: {true_labels[i]} guess: {predicted_labels[i]}")

    return {
        'MAE': mae,
        'MSE': mse,
        'RMSE': rmse,
        'R2 Score': r2
    }

# Main training loop
def train_loop(train_dataloader, eval_dataloader, test_dataloader, num_voxels, model_path, run_mode, y_means= None, y_stds= None):
    """
    The main training loop for training and evaluating the Transformer model.
    Returns:
    tuple: A tuple containing the test accuracy, train loss, and train accuracy.
    """
    global num_heads, learning_rate, epochs, batch_size, dropout, weight_decay, embedding_dim
    model = nn.Sequential(
        TransformerEncoder(num_voxels=num_voxels, classes=1, time2vec_dim=1, num_heads=num_heads,
                           head_size=embedding_dim, ff_dim=embedding_dim,
                           num_layers=3,
                           dropout=dropout
                           )
    )
    model = model.float()
    device = torch.device('cuda:0')
    model.to(device)
    print('cuda: ', torch.cuda.is_available())

    best_loss = 100
    best_train_loss = 100  # To store the best train loss in cross_val mode
    # Defining loss function and optimizer
    class WeightedMSELoss(nn.Module):
        def __init__(self, std):
            super().__init__()
            self.std = torch.tensor(std).float().cuda()

        def forward(self, pred, target):
            squared_error = (pred - target) ** 2
            weighted_error = squared_error / (self.std ** 2)
            return torch.mean(weighted_error)

    # std = torch.mean(torch.stack(y_stds), dim=0)
    # loss_fn = WeightedMSELoss(std)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
   # lr_scheduler = StepLR(optimizer, step_size=3, gamma=0.9) #for learning rate decay

    wandb.watch(model)

    model.train()
    train_loss_ultra = []
    val_loss_ultra = []
    for epoch in range(epochs):


        train_pds, train_gts = [], []
        optimizer.zero_grad()

        # Get one batch of data from the train DataLoader
        first_batch = next(iter(train_dataloader))

        # Unpack the batch (assuming it contains input data and labels)
        inputs_batch, labels_batch = first_batch

        # Print shapes

        with torch.set_grad_enabled(True):
           for idx, train_batch in enumerate(train_dataloader):
                data = train_batch[0].cuda().float()
                data = data.double()
                data = data.cuda()
                gt = train_batch[1].cuda().float()

                train_losses = []

                outputs = model(data.float())  #,y

                train_gts.extend(gt.cpu().detach().numpy().tolist())
                train_pds.extend(outputs.cpu().detach().numpy().tolist())

                loss = loss_fn(outputs, gt)
                train_losses.append(loss.item())

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
               # lr_scheduler.step()

                if idx % 100 == 0:
                    wandb.log({'Train/loss': loss.item(),
                               'Train/epoch': epoch,
                               'Train/step': idx})

           train_loss_ultra.append(sum(train_losses) / len(train_losses))
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

        for metric_name, metric_value in calc_metrics(train_pds, train_gts).items():
            wandb.log( { f'Train/{metric_name}': metric_value})

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
                eval_output = model(eval_data.float()) #,y

                eval_gts.extend(eval_gt.cpu().detach().numpy().tolist())
                eval_pds.extend(eval_output.cpu().detach().numpy().tolist())

                eval_metrics = calc_metrics(eval_pds, eval_gts)
                wandb.log({'Eval/R2_per_epoch': eval_metrics['R2 Score'], 'Eval/epoch': epoch})

                eval_loss = loss_fn(eval_output, eval_gt)
                eval_losses.append(eval_loss.item())

                if idx_val % 100 == 0:
                    wandb.log({'Eval/loss': eval_loss,
                               'Eval/step': idx_val})

            current_eval_loss = sum(eval_losses) / len(eval_losses)
            val_loss_ultra.append(current_eval_loss)
            if current_eval_loss < best_loss:
                torch.save({"model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict()
                            }, model_path)
                best_loss = current_eval_loss
                print(f"Saving best model with eval loss: {best_loss:.4f}")
            if not os.path.exists(model_path):
                torch.save({"model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict()
                            }, model_path)
                print("No best model found, saving last model as fallback.")
            print(f"Eval Loss: {best_loss}")


            # calculating accuracy
            for metric in calc_metrics(eval_pds, eval_gts).items():
                wandb.log({f'Eval/{metric[0]}': metric[1]})
        print(f"First output: {outputs[0].detach().cpu().numpy()}, label: {gt[0].detach().cpu().numpy()}")

    test_losses = []
    test_pds, test_gts = [], []
    print('Testing')
    test_model = model
    test_model = safe_load_model(model_path, test_model, num_voxels)
    test_model.to(device)
    for idx_test, test_batch in enumerate(test_dataloader):
        test_model.eval()
        test_data= test_batch[0].float().cuda()
        test_gt = test_batch[1].cuda().float()
        test_output = test_model(test_data)

        test_gts.extend(test_gt.cpu().detach().numpy().tolist())
        test_pds.extend(test_output.cpu().detach().numpy().tolist())
        test_loss = loss_fn(test_output, test_gt)
        test_losses.append(test_loss.item())
    # === DENORMALIZATION אחרי TEST ===
    # בדיקות ישירות בלי denormalization
    test_gts = np.array(test_gts)
    test_pds = np.array(test_pds)

    print("\n Regression Results by Metric:")
    for i, metric in enumerate(calc_metrics_per_output(test_pds, test_gts)):
        print(
            f"Metric {i + 1}: MAE={metric['MAE']:.2f}, MSE={metric['MSE']:.2f}, RMSE={metric['RMSE']:.2f}, R2={metric['R2']:.2f}")

    print(f"Test Loss: {sum(test_losses) / len(test_losses)}")
    print("\n Regression Results by Metric:")
    for i, metric in enumerate(calc_metrics_per_output(test_pds, test_gts)):
        print(f"Metric {i + 1}: MAE={metric['MAE']:.2f}, MSE={metric['MSE']:.2f}, RMSE={metric['RMSE']:.2f}, R2={metric['R2']:.2f}")

    print(f"Test Loss: {sum(test_losses) / len(test_losses)}")
    wandb.log({'Test/loss': sum(test_losses) / len(test_losses)})
    for metric in calc_metrics(test_pds, test_gts, True).items():
        wandb.log({f'Test/{metric[0]}': metric[1]})

    # logging results to wandb
    if idx_test % 100 == 0:
        wandb.log({'Test/loss': test_loss,
                   'Test/step': idx_test})

    test_metrics = calc_metrics(test_pds, test_gts)
    train_metrics = calc_metrics(train_pds, train_gts)
    wandb.log({'Train/R2_per_epoch': train_metrics['R2 Score'], 'Train/epoch': epoch})
    # === גרפים ===
    results = calc_metrics_per_output(test_pds, test_gts)
    num_outputs = len(results)
    metrics = ['MAE', 'MSE', 'RMSE', 'R2']

    for metric in metrics:
        values = [res[metric] for res in results]
        plt.figure()
        plt.bar(range(1, num_outputs + 1), values)
        plt.xlabel("Output Index")
        plt.ylabel(metric)
        plt.title(f"{metric} per Output")
        plt.grid(True)
        plt.xticks(range(1, num_outputs + 1))
        plt.tight_layout()
        plt.savefig(f"{metric}_per_output.png")
        plt.show()
        wandb.log({f"{metric}_plot": wandb.Image(f"{metric}_per_output.png")})
    x_values = np.arange(1, epochs + 1)
    plt.figure(figsize=(8, 6))
    plt.plot(x_values, train_loss_ultra, label='Train Loss', marker='o')
    plt.plot(x_values, val_loss_ultra, label='Validation Loss', marker='s')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training and Validation Loss For Vis Area Predict One TR')
    plt.legend()
    plt.grid(True)
    plt.show()

    return test_metrics['R2 Score'], sum(train_losses) / len(train_losses), train_metrics['R2 Score']

# Cross-validation function
def run_cross_validation(df_7T):
    """
    Run k-fold cross-validation on the dataset and evaluate model performance.
    Returns:
    list: List of test accuracies for each fold.
    """
    #phase_path = osp.join(directory, 'cross_val', task)
    phase_path=osp.join(directory)
    subject_folders = sorted(glob.glob(phase_path + '/**'))

    # Remove existing SYNTH_ and GROUP_ directories before running cross-validation
    for folder in subject_folders:
        if 'SYNTH_' in folder or 'GROUP_' in folder:
            print(f"Removing synthetic/group subject folder: {folder}")
            for file in glob.glob(osp.join(folder, '*')):
                os.remove(file)
            os.rmdir(folder)
    subject_folders = [folder for folder in subject_folders if 'SYNTH_' not in folder and 'GROUP_' not in folder and os.path.basename(folder) in df_7T["Subject"].astype(str).unique() ]

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
        # df_7T was task
        train_dataloader, num_voxels = get_dataloaders2_for_cross_val(train_files, NET, NET_idx, H, batch_size, slice, df_7T, Avg,
                                                                   noise_level, n_synthetic_TRs, Group_by_subjects, group_size, run_suffix)
        test_dataloader, _ = get_dataloaders2_for_cross_val(test_files, NET, NET_idx, H, batch_size, slice, df_7T, Avg,
                                                         noise_level, n_synthetic_TRs, Group_by_subjects, group_size, run_suffix)

        if train_dataloader is None or test_dataloader is None:
            print(f"Skipping fold {fold + 1} due to missing data.")
            continue

        model_path = f'models/best_model_fold_{fold + 1}.pth'

        fold_accuracy, train_loss, train_accuracy = train_loop(train_dataloader, None, test_dataloader, num_voxels, model_path, 'cross_val',y_means= None, y_stds= None)

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

    train_dataloader, eval_dataloader, test_dataloader, num_voxels, y_means, y_stds = get_dataloaders2_random_split(
        directory, NET, NET_idx, H, slice, batch_size, df_7T,
        Avg=Avg, noise_level=noise_level, n_synthetic_TRs=n_synthetic_TRs,
        Create_synthetic_subjects=Create_synthetic_subjects, n_synthetic_subjects=n_synthetic_subjects,
        Group_by_subjects=Group_by_subjects, group_size=group_size
    )

    def diagnose_dataset(train_loader, y_means, y_stds, max_samples=5):
        print("=== Dataset Diagnostic ===")
        x_list = []
        y_norm_list = []

        for i, (x, y) in enumerate(train_loader):
            if i >= max_samples:
                break
            x_list.append(x)
            y_norm_list.append(y)

        x_batch = torch.cat(x_list, dim=0)  # shape: [max_samples * batch_size, features]
        y_batch = torch.cat(y_norm_list, dim=0)  # shape: [max_samples * batch_size, outputs]

        print(f"[+] Input stats:")
        print(f"Mean of inputs: {x_batch.mean(dim=0)[:5]} ...")
        print(f"Std of inputs: {x_batch.std(dim=0)[:5]} ...")
        print(f"Shape of one input: {x_batch[0].shape}")

        print(f"\n[+] Normalized labels (first 5):\n{y_batch[:5]}")

        means = np.stack([y.numpy() for y in y_means])
        stds = np.stack([s.numpy() for s in y_stds])
        mean = np.mean(means, axis=0)
        std = np.mean(stds, axis=0)
        std[std < 1e-6] = 1.0

        print(
            "\n[✓] Checked variance in inputs and label range. If labels are similar or inputs are flat, model won't learn.")

    # בתוך train():
    diagnose_dataset(train_dataloader, y_means, y_stds)
    if train_dataloader is None or eval_dataloader is None or test_dataloader is None:
        print(f"Skipping {H}_{NET}_{NET_idx} as files are missing.")
        sys.exit()

    wandb.login(
        key = "31edb666543b4d2bf9a513ea5b01e2f87e76867e",
        host="https://api.wandb.ai")
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
    train_loop(train_dataloader, eval_dataloader, test_dataloader, num_voxels, model_path, run_mode, y_means, y_stds)
    wandb.finish()

# Loop for running normal training or cross-validation based on configuration

""""
# Load the data
df = pd.read_csv("all_data.csv")

# Filter the data where '7T_RS-fMRI_Count' is 4
df_7T = df.loc[df['7T_RS-fMRI_Count'] == 4]

# Reset the index and assign it back to df_7T
df_7T = df_7T.reset_index(drop=True)

# Check the information of the filtered DataFrame
df_7T.info()

# Convert the 'Language_Task_Story_Acc' column to a numpy array
y = df_7T['Language_Task_Story_Acc'].to_numpy()
"""






for NET in NET_list:
    for NET_idx in NET_indexes:
        for H in H_list:
            print(f"===> checking {NET}_{NET_idx}_{H} ")
            if f'{NET}_{NET_idx}_{H}' not in exists_list:
                continue
            print(
            f"Running training on {H}_{NET}_{NET_idx} for {epochs} epochs - Batch Size: {batch_size}, Learning Rate: {learning_rate}")
            if run_mode == 'cross_val':
                run_cross_validation(df_7T)
            else:
                train()