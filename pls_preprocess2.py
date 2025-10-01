#preprocess
import glob
import pickle
import pandas as pd
import torch
import os.path as osp
import os
import numpy as np
import re
from sklearn.decomposition import PCA

import random


def organizer(data_files, apply_pca=False, n_components=50, return_mean=False,
              agg='rms', last_n_per_y=None, test=None, i=None):
    all_voxels_per_file = []
    metadata_per_file = []

    for file_path in data_files:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)

        if test == "movies":
            if i == None:
                # Handle NumPy array input (e.g. shape [T, F])
                if not isinstance(data, np.ndarray):
                    raise TypeError(f"[Movies] Expected NumPy array in {file_path}, got {type(data)}")
                voxel_data = pd.DataFrame(data)  # no metadata
                meta_data = pd.DataFrame()
            else:
                if not isinstance(data, pd.DataFrame):
                    raise TypeError(f"File {file_path} is not a DataFrame.")

                if (last_n_per_y is not None) and ('y' in data.columns):
                    data = (data.groupby('y', group_keys=False)
                            .apply(lambda g: g.iloc[-last_n_per_y:] if len(g) > last_n_per_y else g)
                            .reset_index(drop=True))
                if 'y' in data.columns:
                    data = data[data['y'] == i]
                voxel_data = data.drop(columns=['Subject', 'timepoint', 'y'], errors='ignore')
                meta_cols = [c for c in ['Subject', 'timepoint', 'y'] if c in data.columns]
                meta_data = data[meta_cols].reset_index(drop=True)

        else:
            # Original DataFrame format
            if not isinstance(data, pd.DataFrame):
                raise TypeError(f"File {file_path} is not a DataFrame.")

            if (last_n_per_y is not None) and ('y' in data.columns):
                data = (data.groupby('y', group_keys=False)
                        .apply(lambda g: g.iloc[-last_n_per_y:] if len(g) > last_n_per_y else g)
                        .reset_index(drop=True))

            voxel_data = data.drop(columns=['Subject', 'timepoint', 'y'], errors='ignore')
            meta_cols = [c for c in ['Subject', 'timepoint', 'y'] if c in data.columns]
            meta_data = data[meta_cols].reset_index(drop=True)

        all_voxels_per_file.append(voxel_data.reset_index(drop=True))
        metadata_per_file.append(meta_data.reset_index(drop=True))

    if not all_voxels_per_file:
        return None

    features_list = []
    trimmed_meta_first = None

    for i, (voxels_df, meta_df) in enumerate(zip(all_voxels_per_file, metadata_per_file)):
        X = voxels_df.to_numpy()
        meta_trimmed = meta_df.copy()

        col_std = X.std(axis=0, ddof=1)
        keep = col_std > 0
        X = X[:, keep]
        if X.size == 0:
            raise ValueError(f"All-constant columns after filtering in file index {i}")
        #z-score
        X = (X - X.mean(axis=0)) / X.std(axis=0, ddof=1)

        if apply_pca:
            k = int(max(1, min(n_components, X.shape[0] - 1, X.shape[1])))
            scores = PCA(n_components=k, svd_solver='randomized', random_state=0).fit_transform(X)
        else:
            if len(data_files) != 1:
                scores = X.mean(axis=1, keepdims=True)
            else:
                scores = X

        base = osp.splitext(osp.basename(data_files[i]))[0]
        score_cols = [f"{base}__f{j + 1}" for j in range(scores.shape[1])]
        features_list.append(pd.DataFrame(scores, columns=score_cols))

        if trimmed_meta_first is None:
            trimmed_meta_first = meta_trimmed.reset_index(drop=True)

    features_df = pd.concat(features_list, axis=1).reset_index(drop=True)
    meta_df = trimmed_meta_first.reset_index(drop=True)

    if return_mean:
        S = features_df.to_numpy(dtype=np.float32)
        if agg == 'rms':
            feat_vec = np.sqrt((S ** 2).mean(axis=0))
        elif agg == 'mean_abs':
            feat_vec = np.mean(np.abs(S), axis=0)
        elif agg == 'std':
            feat_vec = S.std(axis=0, ddof=1)
        else:
            raise ValueError(f"Unknown agg '{agg}' (use 'rms'/'mean_abs'/'std')")
        return feat_vec

    final_data = pd.concat([features_df, meta_df], axis=1)

    return final_data


def organizer_rest(dataframes, apply_pca=False, n_components=50,
                   return_mean=False, agg='rms', i=None):
    all_voxels_per_file = []
    metadata_per_file = []

    for df in dataframes:
        if isinstance(df, dict):
            df = pd.DataFrame(df)
        elif not isinstance(df, pd.DataFrame):
            raise TypeError("Expected a DataFrame or dict in memory.")

        # Drop the 'is_rest' column if still there
        df = df.drop(columns=['is_rest'], errors='ignore')

        # ---- Per-y filtering logic ----
        if (i is not None) and ('y' in df.columns):
            df = df[df['y'] == i]

        # Keep only voxel data
        voxel_data = df.drop(columns=['Subject', 'timepoint', 'y'], errors='ignore')
        all_voxels_per_file.append(voxel_data.reset_index(drop=True))

        # Metadata
        meta_cols = [c for c in ['Subject', 'timepoint', 'y'] if c in df.columns]
        metadata_per_file.append(df[meta_cols].reset_index(drop=True))
    if not all_voxels_per_file:
        return None
    features_list = []
    trimmed_meta_first = None

    for idx, (voxels_df, meta_df) in enumerate(zip(all_voxels_per_file, metadata_per_file)):
        X = voxels_df.to_numpy()
        meta_trimmed = meta_df.copy()

        # Remove constant columns
        col_std = X.std(axis=0, ddof=1)
        keep = col_std > 0
        X = X[:, keep]
        if X.size == 0:
            raise ValueError(f"All-constant columns after filtering in df index {idx}")

        # Normalize
        mu, sd = X.mean(axis=0), X.std(axis=0, ddof=1)
        sd[sd < 1e-8] = 1.0
        X = (X - mu) / sd
        # PCA or aggregation
        if apply_pca:
            k = int(max(1, min(n_components, X.shape[0] - 1, X.shape[1])))
            scores = PCA(n_components=k, svd_solver='randomized', random_state=0).fit_transform(X)
        else:
            if len(dataframes[0]) != 1:
                scores = X.mean(axis=1, keepdims=True)
            else:
                scores = X

        score_cols = [f"file{idx+1}__f{j+1}" for j in range(scores.shape[1])]
        features_list.append(pd.DataFrame(scores, columns=score_cols))

        if trimmed_meta_first is None:
            trimmed_meta_first = meta_trimmed.reset_index(drop=True)

    features_df = pd.concat(features_list, axis=1).reset_index(drop=True)
    meta_df = trimmed_meta_first.reset_index(drop=True)

    if return_mean:
        # Collapse over time → keep DataFrame format
        S = features_df.to_numpy(dtype=np.float32)
        if agg == 'rms':
            vec = np.sqrt((S ** 2).mean(axis=0))
        elif agg == 'mean_abs':
            vec = np.mean(np.abs(S), axis=0)
        elif agg == 'std':
            vec = S.std(axis=0, ddof=1)
        else:
            raise ValueError(f"Unknown agg '{agg}' (use 'rms'/'mean_abs'/'std')")

        # Return as single-row DataFrame with metadata
        mean_df = pd.DataFrame([vec], columns=features_df.columns)
        final_data = pd.concat([mean_df, meta_df.iloc[[0]].reset_index(drop=True)], axis=1)
        return final_data

    # Default case: keep all timepoints
    final_data = pd.concat([features_df, meta_df], axis=1)
    return final_data



def process_data(df_7T, features, subject_id):
    subject_row = df_7T.loc[df_7T["Subject"] == int(subject_id)].reset_index(drop=True)

    if subject_row.empty:
        raise ValueError(f"Subject ID {subject_id} not found in dataset")
    if features == "fluid intelligence":
        subset_df = subject_row[["PMAT24_A_CR", "PMAT24_A_SI", "PMAT24_A_RTCR"]]
        #feature_names = ["Correct Responses", "Skipped Items", "Time For Correct Responses"]
        feature_names = ["PMAT24_A_CR", "PMAT24_A_SI", "PMAT24_A_RTCR"]

    elif features == "personality":
        subset_df = subject_row[["NEOFAC_A", "NEOFAC_O", "NEOFAC_C", "NEOFAC_N", "NEOFAC_E"]]
        #feature_names = ["Agreeableness", "Openness", "Conscientiousness", "Neuroticism", "Extroversion"]
        feature_names = ["NEOFAC_A", "NEOFAC_O", "NEOFAC_C", "NEOFAC_N", "NEOFAC_E"]
    elif features == "fluid intelligence 2":
        subset_df = subject_row[["PMAT24_A_CR", "PMAT24_A_RTCR"]]
        feature_names = ["Correct Responses", "Time For Correct Responses"]

    elif features == "all":
        subset_df = subject_row

    else:
        raise ValueError("Invalid test option")

    subject_values = subset_df.values.astype(np.float32)
    #print(f"Subject {subject_id} original values: {subject_values}")

    return torch.tensor(subject_values), feature_names


def get_data_for_pls(directory, NET,features, df_7T, MEAN, PCA, split_ratio=(0.8, 0.2), test= None, i=None):
    inputs, outputs = [], []

    subject_folders = list(glob.iglob(osp.join(directory, '*')))
    subject_folders = [folder for folder in subject_folders
                       if 'SYNTH_' not in folder and 'GROUP_' not in folder
                       and os.path.basename(folder) in df_7T["Subject"].astype(str).unique()]


    ref_shape = None  # MEAN=True: F ;  MEAN=False: (T,F)

    for subject_folder in subject_folders:
        try:
            subj = osp.basename(subject_folder)

            if NET[-1].isdigit():
                file_paths = glob.glob(osp.join(subject_folder, f'{NET}.pkl')) #for one file
            else:
                file_paths = glob.glob(osp.join(subject_folder, f'*_{NET}_*.pkl')) #for whole network
            #file_paths = [f for f in file_paths if re.search(rf'_{NET}_\d+\.pkl$', osp.basename(f))] #for after NET filter
            if len(file_paths) == 0:
                continue

            if MEAN:
                # וקטור פיצ'רים אחד לנבדק (סוגרים זמן ב-RMS)
                feat_vec = organizer(file_paths, apply_pca=PCA,return_mean=MEAN,
                                     agg='rms', test= test).astype(np.float32)
                if ref_shape is None:
                    ref_shape = feat_vec.shape[0]
                elif feat_vec.shape[0] != ref_shape:
                    print(f"[Warn] {subj}: features {feat_vec.shape[0]} != {ref_shape}. Skipping.")
                    continue
                inputs.append(feat_vec)
            else:
                if i!= None:
                    ts_df = organizer(file_paths, apply_pca=PCA, return_mean=MEAN,
                                      last_n_per_y=2000, test=test, i=i)
                else:
                    ts_df = organizer(file_paths, apply_pca=PCA, return_mean=MEAN,
                                      last_n_per_y=2000, test=test)

                # Single sample (one subject)
                S = ts_df.iloc[:, :-3].to_numpy(dtype=np.float32)

                inputs.append(S)

                label, feature_names = process_data(df_7T, features, subj)
                outputs.append(label.squeeze().numpy().astype(np.float32))


        except Exception as e:
                print(f"Error processing {subject_folder}: {e}")
                continue

    if len(inputs) == 0 or len(outputs) == 0:
        return None, None, None, None, None
    # בניית X
    if MEAN:
        X = np.vstack(inputs).astype(np.float32)      # [N × F]
    else:
        X = np.stack(inputs).astype(np.float32)       # [N × T × F]

    y = np.vstack(outputs).astype(np.float32)         # [N × targets]

    # --- ערבוב יציב לפני הפיצול ---
    total = X.shape[0]
    rng = np.random.default_rng(42)
    perm = rng.permutation(total)
    X, y = X[perm], y[perm]

    # --- פיצול ---
    train_end = int(split_ratio[0] * total)

    X_train, y_train = X[:train_end], y[:train_end]
    X_test, y_test = X[train_end:], y[train_end:]

    # --- נרמול לפי TRAIN בלבד ---
    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test, feature_names


def get_data_for_pls_rest(directory, NET, features, df_7T, MEAN, PCA, split_ratio=(0.8, 0.2), i=None):
    inputs, outputs = [], []

    if NET[-1].isdigit():
        file_paths = glob.glob(os.path.join(directory, f'{NET}.pkl'))  # exact file
    else:
        file_paths = glob.glob(os.path.join(directory, f'*_{NET}_*.pkl'))  # matching pattern

    ref_shape = None

    for file_path in file_paths:
        with open(file_path, 'rb') as file:
            data = pickle.load(file)
        if isinstance(data, dict):
            df = pd.DataFrame(data)
        elif isinstance(data, pd.DataFrame):
            df = data
        else:
            raise TypeError(f"{file_path} did not contain a dictionary or DataFrame as expected.")

        df = df[df["is_rest"] == 1].copy()
        df.drop("is_rest", axis=1, inplace=True)

        if df.empty:
            print(f"[SKIP] {file_path}: no resting rows.")
            continue

        for subj in df["Subject"].unique():
            if subj not in df_7T["Subject"].values:
                continue

            subj_df = df[df["Subject"] == subj].copy()

            # Instead of saving to file, we monkey-patch the organizer call
            try:

                if i is not None:
                    ts_df = organizer_rest([subj_df], apply_pca=PCA, return_mean=MEAN, i=i)
                else:
                    ts_df = organizer_rest([subj_df], apply_pca=PCA, return_mean=MEAN)



                S = ts_df.iloc[:, :-3].to_numpy(dtype=np.float32)
                inputs.append(S)

                label, feature_names = process_data(df_7T, features, subj)
                outputs.append(label.squeeze().numpy().astype(np.float32))

            except Exception as e:
                print(f"[Error] Subject {subj} in {file_path}: {e}")
                continue
    if len(inputs) == 0 or len(outputs) == 0:
        return None, None, None, None, None

    X = np.vstack(inputs) if MEAN else np.stack(inputs)
    y = np.vstack(outputs).astype(np.float32)

    rng = np.random.default_rng(42)
    perm = rng.permutation(len(X))
    X, y = X[perm], y[perm]

    train_end = int(split_ratio[0] * len(X))
    X_train, y_train = X[:train_end], y[:train_end]
    X_test, y_test = X[train_end:], y[train_end:]

    mean = X_train.mean(axis=0, keepdims=True)
    std = X_train.std(axis=0, keepdims=True)
    std[std < 1e-6] = 1.0

    X_train = (X_train - mean) / std
    X_test = (X_test - mean) / std

    return X_train, y_train, X_test, y_test, feature_names



