import glob
import pickle
import pandas as pd
import torch
import os.path as osp
import os
import numpy as np
from torch.utils.data import Dataset, DataLoader, TensorDataset
import random
from sklearn.decomposition import PCA

# Function to perform z-score normalization
# This function normalizes the data by subtracting the mean and dividing by the standard deviation.
# It helps in scaling the data to have a mean of 0 and a standard deviation of 1.
def z_score_normalize(data, axis):
    return (data - np.mean(data, axis=axis)) / np.std(data, axis=axis)

# Function to perform z-score normalization for series data
# This function applies z-score normalization specifically for series data.
def z_score_normalize_series(data):
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def shuffle_labels(labels):
    np.random.shuffle(labels)
    return labels

# Function to shuffle a DataFrame
# Randomly shuffles both the rows and columns of a DataFrame.
def shuffle_df(df):
    matrix = df.to_numpy()
    np.random.shuffle(matrix)
    matrix = matrix.T
    np.random.shuffle(matrix)
    matrix = matrix.T
    return pd.DataFrame(matrix, columns=df.columns)

# Function to add Gaussian noise to the data
# This function adds Gaussian noise to the input data to generate synthetic samples.
# The noise is controlled by the noise_level parameter, and multiple synthetic samples can be generated.
def add_gaussian_noise(data, noise_level=0.01, n_synthetic=3):
    synthetic_samples = []
    for _ in range(n_synthetic):
        noise = np.random.randn(*data.shape) * noise_level
        synthetic_samples.append(data + noise)
    return synthetic_samples

# Function to average voxel data from multiple files
# This function reads multiple pickle files containing voxel data, averages the voxel values,
# and normalizes the result using z-score normalization.
# The function expects the data in the files to be in a pandas DataFrame format.

def average_voxels(data_files, apply_pca=True, n_components=50):
    all_voxels = []
    metadata = []
    sum = 0
    for file_path in data_files:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            if not isinstance(data, pd.DataFrame):
                raise TypeError("Loaded data is not a DataFrame. Check the file format.")

            for y_value in sorted(data['y'].unique()):
                movie_data = data[data['y'] == y_value]
                voxel_data = movie_data.drop(columns=['Subject', 'timepoint', 'y'], errors='ignore')

                # keep only the last 20 TRs
                if len(voxel_data) >= 20:
                    voxel_data = voxel_data.iloc[-20:, :]

                    meta = movie_data[['Subject', 'timepoint', 'y']].iloc[-20:, :]
                else:
                    print(f"Warning: movie {y_value} in file {file_path} has only {len(voxel_data)} TRs")
                    meta = movie_data[['Subject', 'timepoint', 'y']]

                all_voxels.append(voxel_data)
                metadata.extend(meta.values)

    if not all_voxels:
        return None

    concatenated_data = pd.concat(all_voxels, axis=0).values  # [total_TRs, num_voxels]
    normalized_data = z_score_normalize(concatenated_data, axis=0)

    if apply_pca:
        pca = PCA(n_components=n_components)
        reduced_data = pca.fit_transform(normalized_data)
    else:
        reduced_data = normalized_data

    reduced_df = pd.DataFrame(reduced_data)
    metadata_df = pd.DataFrame(metadata, columns=["Subject", "timepoint", "y"])
    final_data = pd.concat([reduced_df, metadata_df.reset_index(drop=True)], axis=1)
#    print(final_data)
    return final_data
# Function to average the last few TRs (time points) for each movie and add synthetic samples
# This function averages the last few time points (TRs) for each movie in the dataset.
# It also adds synthetic TRs by introducing Gaussian noise, and optionally applies PCA for dimensionality reduction.
def average_TRs(data_files, noise_level=0.01, n_synthetic_TRs=3):   # for PCA test use n_components=10
    averaged_trs = {i: [] for i in range(15)}
    metadata_list = {i: [] for i in range(15)}

    for file_path in data_files:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            if isinstance(data, pd.DataFrame):
                for y_value in range(15):
                    movie_data = data[data['y'] == y_value]
                    voxel_data = movie_data.drop(columns=['Subject', 'timepoint', 'y'], errors='ignore')
                    metadata = movie_data.iloc[:, -3:]

                    last_TRs = voxel_data.iloc[-5:, :]
                    mean_TRs = last_TRs.mean(axis=0)
                    averaged_trs[y_value].append(mean_TRs)

                    synthetic_samples = add_gaussian_noise(mean_TRs, noise_level, n_synthetic_TRs)
                    for synthetic_data in synthetic_samples:
                        averaged_trs[y_value].append(synthetic_data)
                        metadata_list[y_value].append(metadata.iloc[-1, :])

                    metadata_last = metadata.iloc[-1, :]
                    metadata_list[y_value].append(metadata_last)
            else:
                raise TypeError("Loaded data is not a DataFrame. Check the file format.")

    final_data_list = []
    for y_value in range(15):
        concatenated_trs = np.concatenate(averaged_trs[y_value])
        final_data_list.append(concatenated_trs)

    final_data = pd.DataFrame(final_data_list)
    normalized_final_data = z_score_normalize(final_data, axis=0)

    # Uncomment the following block to apply PCA for dimensionality reduction
    """
    # Apply PCA to reduce dimensionality
    pca = PCA(n_components=n_components)
    pca_result = pca.fit_transform(normalized_final_data)

    # Convert PCA result to DataFrame
    pca_df = pd.DataFrame(pca_result)
    """

    final_metadata_list = []
    for y_value in range(15):
        averaged_metadata = pd.DataFrame(metadata_list[y_value]).mean(axis=0).to_frame().T

        if 'timepoint' in averaged_metadata.columns:
            averaged_metadata['timepoint'] = averaged_metadata['timepoint'].astype(int)
        if 'Subject' in averaged_metadata.columns:
            averaged_metadata['Subject'] = averaged_metadata['Subject'].astype(int)

        final_metadata_list.append(averaged_metadata)

    metadata_df = pd.concat(final_metadata_list, ignore_index=True)
    metadata_df['y'] = range(15)

    final_data = pd.concat([normalized_final_data, metadata_df.reset_index(drop=True)], axis=1)   # if PCA was used - use pca_df instead of normalized_final_data

    return final_data

def process_data(test, subject_id):
    df = pd.read_csv("all_data.csv")
    df_7T = df.loc[df['7T_RS-fMRI_Count'] == 4].reset_index(drop=True)
    subject_row = df_7T.loc[df_7T["Subject"] == int(subject_id)].reset_index(drop=True)

    if subject_row.empty:
        raise ValueError(f"Subject ID {subject_id} not found in dataset")

    if test == "Default_PFC":
        subset_df = subject_row[["PMAT24_A_CR", "PMAT24_A_SI", "PMAT24_A_RTCR"]]
        all_labels = df_7T[["PMAT24_A_CR", "PMAT24_A_SI", "PMAT24_A_RTCR"]].values.astype(np.float32)
    elif test == "2":
        pass
    elif test == "3":
        subset_df = subject_row[["NEOFAC_A", "NEOFAC_O", "NEOFAC_C", "NEOFAC_N", "NEOFAC_E"]]
        all_labels = df_7T[["NEOFAC_A", "NEOFAC_O", "NEOFAC_C", "NEOFAC_N", "NEOFAC_E"]].values.astype(np.float32)
    elif test == "all":
        # תשלימי בעתיד
        pass
    else:
        raise ValueError("Invalid test option")

    mean = all_labels.mean(axis=0)
    std = all_labels.std(axis=0)
    std[std < 1e-6] = 1.0

    #print(f"Global Mean: {mean}")
    #print(f"Global Std: {std}")

    subject_values = subset_df.values.astype(np.float32)
    #print(f"Subject {subject_id} original values: {subject_values}")

    normalized = (subject_values - mean) / std
    #print(f"Normalized values: {normalized}\n")
    return torch.tensor(normalized), mean, std


# Function to process a directory and average voxel data based on the configuration
# This function processes all the relevant files in a directory, averages the voxel data if specified

def process_directory(subject_folder, H, NET, NET_idx, Avg, noise_level=0.01, n_synthetic_TRs=3):
    try:
        if H == 'BOTH':
            files = glob.glob(osp.join(subject_folder, 'LH_' + NET + '*.pkl')) + \
                    glob.glob(osp.join(subject_folder, 'RH_' + NET + '*.pkl'))
            data_files = [file for file in files if not file.endswith('Avg.pkl')]
        else:
            data_files = glob.glob(osp.join(subject_folder, H + '_' + NET + '_' + str(NET_idx) + '.pkl'))
        if not data_files:
            print(f"No Default files found in {subject_folder}.")
            return

        if Avg == 1:
            averaged_data = average_voxels(data_files, apply_pca=True, n_components=50)
            print("shape of averaged", averaged_data.shape)
            if averaged_data is None:
                print("No valid data to average.")
                return
        elif Avg == 2:
            averaged_data = average_TRs(data_files, noise_level=noise_level, n_synthetic_TRs=n_synthetic_TRs)
            if averaged_data is None:
                print("No valid data to average.")
                return

        if H == 'BOTH':
            output_file = osp.join(subject_folder, H + '_' + NET + '_Avg.pkl')
        else:
            output_file = osp.join(subject_folder, H + '_' + NET + '_' + str(NET_idx) + '_Avg.pkl')
        if os.path.exists(output_file):
            os.remove(output_file)
        with open(output_file, 'wb') as f:
            pickle.dump(averaged_data, f)
    except Exception as e:
        print(f"Error processing directory {subject_folder}: {e}")

# Function to create synthetic subjects with Gaussian noise
# This function generates synthetic subjects by adding Gaussian noise to the original data.
# It creates new folders for each synthetic subject and saves the modified data there.
def create_synthetic_subjects(subject_folder, H, NET, NET_idx, noise_level=0.01, n_synthetic_subjects=10):
    if 'SYNTH_' in subject_folder or 'GROUP_' in subject_folder:
        return []

    synthetic_folders = []
    subject_id = osp.basename(subject_folder)

    for i in range(n_synthetic_subjects):
        current_noise_level = noise_level * (i + 1)
        synthetic_folder = f"{subject_folder}_SYNTH_{i + 1}"
        synthetic_folders.append(synthetic_folder)
        os.makedirs(synthetic_folder, exist_ok=True)

        if H == 'BOTH':
            files = glob.glob(osp.join(subject_folder, 'LH_' + NET + '*.pkl')) + \
                    glob.glob(osp.join(subject_folder, 'RH_' + NET + '*.pkl'))
        else:
            files = glob.glob(osp.join(subject_folder, H + '_' + NET + '_' + str(NET_idx) + '.pkl'))

        for file_path in files:
            with open(file_path, 'rb') as file:
                data = pickle.load(file)
                if isinstance(data, pd.DataFrame):
                    voxel_data = data.iloc[:, :-3]
                    metadata = data.iloc[:, -3:]

                    synthetic_voxel_data = add_gaussian_noise(voxel_data.to_numpy(), current_noise_level, n_synthetic=1)[0]
                    synthetic_data = pd.concat([pd.DataFrame(synthetic_voxel_data, columns=voxel_data.columns), metadata], axis=1)

                    synthetic_file_path = osp.join(synthetic_folder, osp.basename(file_path))
                    with open(synthetic_file_path, 'wb') as syn_file:
                        pickle.dump(synthetic_data, syn_file)
                else:
                    raise TypeError("Loaded data is not a DataFrame. Check the file format.")

    return synthetic_folders

# Function to get dataloaders for training, evaluation, and testing (called on ly in normal training, not in cross validation)
def get_dataloaders2(directory, NET, NET_idx, H, slice, batch_size, df_7T, Avg=0, noise_level=0.01,
                     n_synthetic_TRs=0, Create_synthetic_subjects=0, n_synthetic_subjects=0, Group_by_subjects=0,
                     group_size=10, use_original_for_val_test=False, fold=None):
    """
    Prepares dataloaders for training, evaluation, and testing by loading and processing fMRI data.
    Returns:
        tuple: Dataloaders for training, validation, and testing, and the number of voxels.

    Functionality:
        - Loads and processes data for different phases (train, eval, test).
        - Can create synthetic subjects by adding noise to original data.
        - Can group subjects together and average their data.
        - Loads the processed data into PyTorch dataloaders for model training and evaluation.
    """

    dataloaders = {}
    file_exists = True
    inputs = {'train': [], 'eval': [], 'test': []}
    outputs = {'train': [], 'eval': [], 'test': []}

    for phase in ['train', 'eval', 'test'] :
        #inputs = []  # Initialize inputs here
        #outputs = []  # Initialize outputs here

        #phase_path = osp.join(directory, phase, task)
        phase_path=osp.join(directory)
        subject_folders = list(glob.iglob(phase_path + '/**'))
        # Remove synthetic subject folders if they exist
        for folder in subject_folders:
            if 'SYNTH_' in folder or 'GROUP_' in folder:
                print(f"Removing synthetic subject folder: {folder}")
                for file in glob.glob(osp.join(folder, '*')):
                    os.remove(file)
                os.rmdir(folder)
        subject_folders = [folder for folder in subject_folders if 'SYNTH_' not in folder and 'GROUP_' not in folder and os.path.basename(folder) in df_7T["Subject"].astype(str).unique() ]

        # Handle synthetic subjects creation
        if Create_synthetic_subjects == 1 and (
                use_original_for_val_test == False or (use_original_for_val_test == True and phase == 'train')):
            for subject_folder in subject_folders:
                synthetic_folders = create_synthetic_subjects(subject_folder, H, NET, NET_idx, noise_level=noise_level,
                                                              n_synthetic_subjects=n_synthetic_subjects)
                subject_folders.extend(synthetic_folders)

        # Handle grouping subjects together
        if Group_by_subjects == 1:
            for folder in subject_folders:
                if 'GROUP_' in folder:
                    print(f"Removing subjects group folder: {folder}")
                    for file in glob.glob(osp.join(folder, '*')):
                        os.remove(file)
                    os.rmdir(folder)
            subject_folders = [folder for folder in subject_folders if 'GROUP_' not in folder]

            random.shuffle(subject_folders)
            group_folders_list = []
            for i in range(0, len(subject_folders), group_size):
                if len(subject_folders[i:i + group_size]) < group_size:
                    break
                group_folders = subject_folders[i:i + group_size]
                group_folder_name = f"GROUP_{osp.basename(group_folders[0])}_{osp.basename(group_folders[-1])}"
                if fold is not None:
                    group_folder_name += f"_fold_{fold}"
                group_folder_path = osp.join(phase_path, group_folder_name)
                os.makedirs(group_folder_path, exist_ok=True)

                for H_ in ['LH', 'RH']:
                    if H_ == H or H == 'BOTH':
                        for net_file in glob.glob(osp.join(group_folders[0], H_ + '_' + NET + '*.pkl')):
                            net_files = [osp.join(folder, osp.basename(net_file)) for folder in group_folders]
                            averaged_data = average_subjects(net_files)
                            output_file = osp.join(group_folder_path, osp.basename(net_file))
                            with open(output_file, 'wb') as f:
                                pickle.dump(averaged_data, f)
                group_folders_list.append(group_folder_path)
            subject_folders = group_folders_list

            for group_folder in group_folders_list:
                process_directory(group_folder, H, NET, NET_idx, Avg, noise_level=noise_level,
                                  n_synthetic_TRs=n_synthetic_TRs)

        # Process data for each subject
        for subject_folder in subject_folders:
            if Avg in [7, 2]:
                process_directory(subject_folder, H, NET, NET_idx, Avg, noise_level=noise_level,
                                  n_synthetic_TRs=n_synthetic_TRs)

            try:
                if Avg in [7, 2]:
                    if H == "BOTH":
                        file_path = osp.join(subject_folder, '{}_{}_{}.pkl'.format(H, NET, 'Avg'))
                    else:
                        file_path = osp.join(subject_folder, '{}_{}_{}_{}.pkl'.format(H, NET, NET_idx, 'Avg'))
                else:
                    file_path = osp.join(subject_folder, '{}_{}_{}.pkl'.format(H, NET, NET_idx))

                if not osp.exists(file_path):
                    print(f"File does not exist: {file_path}")
                    continue


                with open(file_path, 'rb') as file:
                    data_vis = pickle.load(file)
                    print(f"matrix is {data_vis}")


                data_vis = data_vis.drop(columns=['Subject', 'timepoint', 'y'],
                                         errors='ignore')
                num_voxels = data_vis.shape[1]
                subject_label = process_data(NET, os.path.basename(subject_folder))
                outputs[phase].append(subject_label.squeeze())
                inputs[phase].append(torch.tensor(data_vis.values))

            except Exception as e:
                print(f"Error processing file {file_path}: {e}")
            except FileNotFoundError:
                file_exists = False
                continue

        if not file_exists:
            continue

        # Ensure inputs and outputs are consistent before stacking
        if len(inputs) > 0 and len(outputs) > 0:
            tensor_inputs = torch.stack(inputs[phase])
            tensor_inputs = tensor_inputs.double()

            print(f"Shape of stacked inputs in {phase}: {tensor_inputs.shape}")

            labels = torch.stack(outputs[phase])


            # Create dataloaders
            if phase == 'train':
                train_labels = labels
                train_dataset = TensorDataset(tensor_inputs, train_labels)
                train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

                dataloaders[phase] = train_dataloader
            elif phase == 'eval':
                eval_labels = labels
                eval_dataset = TensorDataset(tensor_inputs, eval_labels)
                eval_dataloader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=True)
                dataloaders['val'] = eval_dataloader
            else:
                test_labels = labels
                test_dataset = TensorDataset(tensor_inputs, test_labels)
                test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=True)
                dataloaders[phase] = test_dataloader

    if not file_exists:
        return None, None, None, None

    return dataloaders['train'], dataloaders['val'], dataloaders['test'], num_voxels


# Function to average subject data after grouping several subjects together
def average_subjects(data_files):
    """
    Averages the data of multiple subjects together.
   Returns:
    Pandas DataFrame: A DataFrame containing the averaged data of all subjects.
    """
    averaged_data_list = []
    metadata_list = []

    for file_path in data_files:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
            if isinstance(data, pd.DataFrame):
                voxel_data = data.drop(columns=['Subject', 'timepoint', 'y'], errors='ignore')
                metadata = data.iloc[:, -3:]
                averaged_data_list.append(voxel_data)
                metadata_list.append(metadata)
            else:
                raise TypeError("Loaded data is not a DataFrame. Check the file format.")

    if not averaged_data_list:
        return None

    concatenated_data = pd.concat(averaged_data_list).groupby(level=0).mean()
    concatenated_metadata = pd.concat(metadata_list).groupby(level=0).mean()

    normalized_data = z_score_normalize(concatenated_data, axis=0)

    final_data = pd.concat([normalized_data, concatenated_metadata], axis=1)

    return final_data

# Helper function for cross-validation dataloaders
def get_dataloaders2_helper(data_files, NET, NET_idx, H, slice, batch_size, df_7T, Avg, noise_level, n_synthetic_TRs, Group_by_subjects, group_size, run_suffix):
    """
    Prepares a dataloader for cross-validation by loading and processing fMRI data. This function handles grouping subjects, averaging data, and adding synthetic data if needed.
    Returns:
    tuple: A tuple containing the dataloader and the number of voxels.
    """
    inputs, outputs = [], []
    subject_folders = sorted({
        osp.dirname(file) for file in data_files
        if osp.basename(osp.dirname(file)) in set(df_7T["Subject"].astype(str).unique())
    })

    if Group_by_subjects == 1:
        #random.shuffle(subject_folders)
        group_folders_list = []
        for i in range(0, len(subject_folders), group_size):
            if len(subject_folders[i:i + group_size]) < group_size:
                break
            group_folders = subject_folders[i:i + group_size]
            group_folder_name = f"GROUP_{osp.basename(group_folders[0])}_{osp.basename(group_folders[-1])}_{run_suffix}"
            group_folder_path = osp.join(osp.dirname(group_folders[0]), group_folder_name)
            os.makedirs(group_folder_path, exist_ok=True)

            for H_ in ['LH', 'RH']:
                if H_ == H or H == 'BOTH':
                    for net_file in glob.glob(osp.join(group_folders[0], H_ + '_' + NET + '*.pkl')):
                        net_files = [osp.join(folder, osp.basename(net_file)) for folder in group_folders]
                        averaged_data = average_subjects(net_files)
                        output_file = osp.join(group_folder_path, osp.basename(net_file))
                        with open(output_file, 'wb') as f:
                            pickle.dump(averaged_data, f)
            group_folders_list.append(group_folder_path)
        subject_folders = group_folders_list

        for group_folder in group_folders_list:
            process_directory(group_folder, H, NET, NET_idx, Avg, noise_level=noise_level, n_synthetic_TRs=n_synthetic_TRs)

    for subject_folder in subject_folders:
        if Avg in [7, 2]:
            process_directory(subject_folder, H, NET, NET_idx, Avg, noise_level=noise_level, n_synthetic_TRs=n_synthetic_TRs)

        try:
            if Avg in [7, 2]:
                if H == "BOTH":
                    file_path = osp.join(subject_folder, '{}_{}_{}.pkl'.format(H, NET, 'Avg'))
                else:
                    file_path = osp.join(subject_folder, '{}_{}_{}_{}.pkl'.format(H, NET, NET_idx, 'Avg'))
            else:
                file_path = osp.join(subject_folder, '{}_{}_{}.pkl'.format(H, NET, NET_idx))
            if not osp.exists(file_path):
                print(f"File does not exist: {file_path}")
                continue

            with open(file_path, 'rb') as file:
                data_vis = pickle.load(file)
            num_voxels = data_vis.shape[1]
            #inputs, outputs = process_movie_data(data_vis, slice, task, inputs, outputs)
            #inputs, outputs = process_data(data_vis)
            outputs = process_data(NET)
            inputs = torch.tensor(data_vis.values)



            #print(f"Processed {file_path}, inputs length: {len(inputs)}, outputs length: {len(outputs)}")

        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
        except FileNotFoundError:
            continue

    if len(inputs) == 0 or len(outputs) == 0:
        return None, None

    tensor_inputs = torch.stack(inputs)
    tensor_inputs = tensor_inputs.double()
    labels = torch.stack(outputs)

    dataset = TensorDataset(tensor_inputs, labels)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    return dataloader, num_voxels

# Function to get dataloaders for cross-validation
def get_dataloaders2_for_cross_val(data_files, NET, NET_idx, H, batch_size, slice, df_7T, Avg, noise_level, n_synthetic_TRs, Group_by_subjects, group_size, run_suffix):
    """
    Prepares dataloaders for cross-validation by loading and processing fMRI data. This function includes steps for grouping subjects, averaging data, and adding synthetic samples if required.
    Returns:
    tuple: A tuple containing the dataloader and the number of voxels.
    """

    files = [file for i, file in enumerate(data_files)]
    dataloader, num_voxels = get_dataloaders2_helper(files, NET, NET_idx, H, slice, batch_size, df_7T, Avg, noise_level, n_synthetic_TRs, Group_by_subjects, group_size, run_suffix)
    return dataloader, num_voxels

from torch.utils.data import random_split

def get_dataloaders2_random_split(directory, NET, NET_idx, H, slice, batch_size, df_7T, Avg=0, noise_level=0.01,
                                  n_synthetic_TRs=0, Create_synthetic_subjects=0, n_synthetic_subjects=0,
                                  Group_by_subjects=0, group_size=10, split_ratio=(0.7, 0.15, 0.15)):
    inputs = []
    outputs = []
    y_means = []
    y_stds = []
    subject_folders = list(glob.iglob(osp.join(directory, '*')))
    subject_folders = [folder for folder in subject_folders
                       if 'SYNTH_' not in folder and 'GROUP_' not in folder
                       and os.path.basename(folder) in df_7T["Subject"].astype(str).unique()]

    for subject_folder in subject_folders:
        try:
            if Avg in [7, 2]:
                process_directory(subject_folder, H, NET, NET_idx, Avg, noise_level=noise_level,
                                  n_synthetic_TRs=n_synthetic_TRs)

            if Avg in [7, 2]:
                file_path = osp.join(subject_folder, f'{H}_{NET}_{NET_idx}_Avg.pkl')
            else:
                file_path = osp.join(subject_folder, f'{H}_{NET}_{NET_idx}.pkl')

            if not osp.exists(file_path):
                print(f"File does not exist: {file_path}")
                continue

            if Avg == 1:
                data_vis = average_voxels([file_path], n_components=20)

            else:
                with open(file_path, 'rb') as file:
                    data_vis = pickle.load(file)


            data_vis = data_vis.drop(columns=['Subject', 'timepoint', 'y'], errors='ignore')
            inputs.append(torch.tensor(data_vis.values, dtype=torch.float64))
            label, y_mean, y_std = process_data(NET, os.path.basename(subject_folder))
            outputs.append(label.squeeze())
            y_means.append(torch.tensor(y_mean))
            y_stds.append(torch.tensor(y_std))

        except Exception as e:
            print(f"Error processing {subject_folder}: {e}")
            continue

    if len(inputs) == 0 or len(outputs) == 0:
        return None, None, None, None, None, None

    tensor_inputs = torch.stack(inputs)
    tensor_outputs = torch.stack(outputs)

    global_mean = tensor_inputs.mean(dim=0, keepdim=True)
    global_std = tensor_inputs.std(dim=0, keepdim=True)
    global_std[global_std < 1e-6] = 1.0
    tensor_inputs = (tensor_inputs - global_mean) / global_std

    dataset = TensorDataset(tensor_inputs, tensor_outputs)

    train_size = int(split_ratio[0] * len(dataset))
    val_size = int(split_ratio[1] * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_set, val_set, test_set = random_split(dataset, [train_size, val_size, test_size])
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, tensor_inputs.shape[-1], y_means, y_stds

