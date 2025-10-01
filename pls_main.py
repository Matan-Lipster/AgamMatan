#main
import torch
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import wandb
import time # for timestamp in wandbf
import os
from pls_preprocess2 import get_data_for_pls, get_data_for_pls_rest
from pls_model import pls_run
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd




import torch.nn.functional as F
os.makedirs("models", exist_ok = True)
# Global Hyperparameters

# cross_val hyperparameters
#directory = r"D:\Final Project\TASK_RH_vis2\dataset"
directory = r"F:\HCP_DATA"
# Load the data
df_7T = pd.read_csv("all_data.csv")
# Filter the data where '7T_RS-fMRI_Count' is 4
df_7T = df_7T.loc[df_7T['7T_RS-fMRI_Count'] == 4]
# Reset the index and assign it back to df_7T
df_7T = df_7T.reset_index(drop=True)


test=input("choose from 1:Default_PFC_2, 2:Limbic_OFC + Default_PFC, 3:Default + Vis, 300: sparta legion,\n "
           "rest: all resting state, movies: per movie, all:all is all man: ")
features = "fluid intelligence" #fluid intelligence/personality/ fluid intelligence 2/ all
if test == "1":
    NET_list = ['Limbic']
    #NET_list = ['RH_SalVentAttn_TempOccPar_4', 'LH_SomMot_6', 'RH_Cont_PFCl_1', 'RH_Default_PFCdPFCm_4', 'LH_DorsAttn_Post_6', 'RH_Limbic_TempPole_3', 'RH_Vis_7']
    #NET_list = ['LH_SalVentAttn_FrOperIns_1', 'LH_SomMot_10', 'RH_Cont_PFCv_1', 'RH_Default_PFCv_3', 'LH_DorsAttn_Post_6', 'RH_Limbic_TempPole_3', 'RH_Vis_1']
    #NET_list = [
    #    'RH_Default_pCunPCC_3', 'LH_Default_pCunPCC_4', 'RH_Default_PFCv_2', # NEOFAC_O
    #    'RH_Vis_16', 'RH_Cont_PFCmp_1',  # NEOFAC_C
    #    'LH_Vis_20', 'RH_Default_Par_5', 'RH_Default_pCunPCC_4',  # NEOFAC_N
    #]


elif test == "2":
    NET_list = ['Limbic_OFC', 'Default_PFC']
elif test == "3":
    NET_list = ['Default', 'Vis']
elif test == "300":
    all_nets_path = r"F:\HCP_DATA\100610"
    NET_list = []

    for file in os.listdir(all_nets_path):
        if file.endswith(".pkl"):
            net_name = os.path.splitext(file)[0]  # remove ".pkl"
            NET_list.append(net_name)

    print("NET_list length =", len(NET_list))
elif test == "movies":
    #modirectory = r"F:\HCP_movie_base_norm"
    #directory = r"D:\Final Project\Predicting Human Brain States with Transformer\Gal&Yuval code\Processed_Matrices"
    if directory ==r"D:\Final Project\Predicting Human Brain States with Transformer\Gal&Yuval code\Processed_Matrices":
        all_nets_path = r"D:\Final Project\Predicting Human Brain States with Transformer\Gal&Yuval code\Processed_Matrices\100610"
        NET_list = []

        for file in os.listdir(all_nets_path):
            if file.endswith(".pkl"):
                net_name = os.path.splitext(file)[0]  # remove ".pkl"
                NET_list.append(net_name)

        print("NET_list length =", len(NET_list))
    else:
        #NET_list = ['LH_SalVentAttn_FrOperIns_1', 'LH_SomMot_10', 'RH_Cont_PFCl_1', 'RH_Default_PFCv_3', 'LH_DorsAttn_Post_6', 'RH_Limbic_TempPole_3', 'RH_Vis_9']
        #NET_list = ['Cont', 'Default', 'DorsAttn', 'Limbic', 'SalVentAttn', 'SomMot', 'Vis']
        #NET_list = ['RH_Default_pCunPCC_6', 'RH_Default_pCunPCC_3', 'RH_Cont_PFCl_1', 'RH_Vis_16', 'LH_Vis_20']
        #NET_list = ['LH_SalVentAttn_FrOperIns_1','RH_Default_PFCv_3','RH_Limbic_TempPole_3',
        #            'LH_SomMot_6','LH_DorsAttn_Post_6','LH_Cont_Temp_1','RH_Vis_1']
        NET_list = ['SomMot']
elif test == "rest":
    directory = r"F:\HCP_resting_state_base_norm"
    NET_list = []

    for file in os.listdir(directory):
        if file.endswith(".pkl"):
            net_name = os.path.splitext(file)[0]  # remove ".pkl"
            NET_list.append(net_name)

    print("NET_list length =", len(NET_list))

elif test == "all":
    NET_list = ['Cont', 'Default', 'DorsAttn', 'Limbic', 'SalVentAttn', 'SomMot', 'Vis']
else:
    print("c'mon man.... play along")

MEAN = False
PCA = False
MUTATE = True
timestamp = time.strftime("%d%m-%H%M")  # timestamp for wandb

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
        print(f"{'Output':>6} | {'MAE':>10} | {'MSE':>10} | {'RMSE':>10} | {'R2':>10}")
        print("-" * 60)
        print(f"{i:6} | {mae:10.4f} | {mse:10.4f} | {rmse:10.4f} | {r2:10.4f}")
        print(f"i= {i} y_true= {y_true}  \n y_pred= {y_pred} \n")
        results.append({
            'MAE': mae,
            'MSE': mse,
            'RMSE': rmse,
            'R2': r2
        })
    return results

def permutator(X_train, y_train, X_test, y_test):
    global cor_array  # use the global list
    for j in range(0, 100):
        print("run: ", j)
        if j == 0:
            # First run: no permutation, use original y_train
            y_train_perm = y_train
        else:
            # Subsequent runs: permute y_train
            y_train_perm = np.random.permutation(y_train)

        corr, y_true, y_pred = pls_run(X_train, y_train_perm, X_test, y_test, n_components=5)

        n_targets = y_true.shape[1] if y_true.ndim > 1 else 1
        if n_targets > 1:
            corr_matrix = np.empty((n_targets, n_targets))
            for m in range(n_targets):
                for n in range(n_targets):
                    corr_matrix[m, n] = np.corrcoef(y_true[:, m], y_pred[:, n])[0, 1]

            cor_array.append(np.diag(corr_matrix))
    return y_true, y_pred, cor_array



# Training function for normal train mode
cor_array = []
std_array = []
# Training function for normal train mode
cor_array = []
std_array = []

def train():
    global feature_names, cor_array  # <- make cor_array explicitly global

    if directory == r"F:\HCP_resting_state_base_norm":
        X_train, y_train, X_test, y_test, feature_names = get_data_for_pls_rest(
            directory=directory,
            NET=NET,
            features=features,
            df_7T=df_7T,
            MEAN=MEAN,
            PCA=PCA,
            split_ratio=(0.8, 0.2)
        )
        if MUTATE:
            # DON'T rebind cor_array here; permutator updates the global list
            y_true, y_pred, _ = permutator(X_train, y_train, X_test, y_test)
        else:
            corr, y_true, y_pred = pls_run(X_train, y_train, X_test, y_test, n_components=5)

    else:
        if test == "movies" and directory != r"D:\Final Project\Predicting Human Brain States with Transformer\Gal&Yuval code\Processed_Matrices":
            if directory == r"F:\HCP_movie_base_norm":
                for i in range(1, 15):  # no movie 0 in these files
                    X_train, y_train, X_test, y_test, feature_names = get_data_for_pls_rest(
                        directory=directory,
                        NET=NET,
                        features=features,
                        df_7T=df_7T,
                        MEAN=MEAN,
                        PCA=PCA,
                        split_ratio=(0.8, 0.2),
                        i=i
                    )
                    if MUTATE:
                        y_true, y_pred, _ = permutator(X_train, y_train, X_test, y_test)
                    else:
                        corr, y_true, y_pred = pls_run(X_train, y_train, X_test, y_test, n_components=5)

                        n_targets = y_true.shape[1] if y_true.ndim > 1 else 1
                        if n_targets > 1:
                            corr_matrix = np.empty((n_targets, n_targets))
                            for m in range(n_targets):
                                for n in range(n_targets):
                                    corr_matrix[m, n] = np.corrcoef(y_true[:, m], y_pred[:, n])[0, 1]
                            cor_array.append(np.diag(corr_matrix))
            else:
                #for i in range(0, 15):
                    X_train, y_train, X_test, y_test, feature_names = get_data_for_pls(
                        directory=directory,
                        NET=NET,
                        features=features,
                        df_7T=df_7T,
                        MEAN=MEAN,
                        PCA=PCA,
                        split_ratio=(0.8, 0.2),
                        test=test,
                        i=5
                    )
                    if MUTATE:
                        y_true, y_pred, _ = permutator(X_train, y_train, X_test, y_test)
                    else:
                        corr, y_true, y_pred = pls_run(X_train, y_train, X_test, y_test, n_components=5)
                        n_targets = y_true.shape[1] if y_true.ndim > 1 else 1
                        if n_targets > 1:
                            corr_matrix = np.empty((n_targets, n_targets))
                            for m in range(n_targets):
                                for n in range(n_targets):
                                    corr_matrix[m, n] = np.corrcoef(y_true[:, m], y_pred[:, n])[0, 1]
                            cor_array.append(np.diag(corr_matrix))
        else:
            X_train, y_train, X_test, y_test, feature_names = get_data_for_pls(
                directory=directory,
                NET=NET,
                features=features,
                df_7T=df_7T,
                MEAN=MEAN,
                PCA=PCA,
                split_ratio=(0.8, 0.2),
                test=test,
                i=None
            )
            if MUTATE:
                y_true, y_pred, _ = permutator(X_train, y_train, X_test, y_test)
            else:
                corr, y_true, y_pred = pls_run(X_train, y_train, X_test, y_test, n_components=5)

    print("\n[+] Evaluation Metrics on Test Set:")
    metrics = calc_metrics_per_output(y_pred, y_true)

    n_targets = y_true.shape[1] if y_true.ndim > 1 else 1
    if n_targets > 1:
        corr_matrix = np.empty((n_targets, n_targets))
        for i in range(n_targets):
            for j in range(n_targets):
                corr_matrix[i, j] = np.corrcoef(y_true[:, i], y_pred[:, j])[0, 1]

        # saving the diagonal cross
        if not (test == "movies" and directory != r"D:\Final Project\Predicting Human Brain States with Transformer\Gal&Yuval code\Processed_Matrices"):
            cor_array.append(np.diag(corr_matrix))

    def diagnose_numpy_dataset(X, y, max_samples=5):
        print("=== Numpy Dataset Diagnostic ===")
        print(f"Shape of X: {X.shape}")
        print(f"Shape of y: {y.shape}")

    diagnose_numpy_dataset(X_train, y_train)

    wandb.login(key="31edb666543b4d2bf9a513ea5b01e2f87e76867e", host="https://api.wandb.ai")


for NET_NUM, NET in enumerate(NET_list):
    print(f"===> checking {NET}")
    train()


# Convert list of correlations to NumPy array
cor_array = np.array(cor_array)
print("cor_array shape:", cor_array.shape)
n_networks, n_features = cor_array.shape

# Use turbo colormap to assign unique color per network "type"
cmap = plt.get_cmap("turbo")

# Extract network category from NET_list (middle part, e.g., Vis from RH_Vis_1)
color_keys = [net.split('_')[1] if '_' in net else net for net in NET_list]
unique_keys = sorted(set(color_keys))
color_map = {key: cmap(i / len(unique_keys)) for i, key in enumerate(unique_keys)}
bar_colors = [color_map[key] for key in color_keys]

if MUTATE:
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    n_networks = len(NET_list)
    n_features = cor_array.shape[1]
    n_perms = cor_array.shape[0] // n_networks

    # reshape once: [nets × perms × features]
    carr = cor_array.reshape(n_networks, n_perms, n_features)

    # consistent colors per network (keep NET_list order)
    cmap = plt.get_cmap("turbo")
    net_colors = {net: cmap(i / max(1, n_networks - 1)) for i, net in enumerate(NET_list)}

    for j in range(n_features):
        feature_name = feature_names[j]

        # observed per-net (perm_idx=0); permuted pool (all others, flattened)
        obs_vals = carr[:, 0, j]                                  # shape [n_networks]
        perm_vals = carr[:, 1:, j].ravel() if n_perms > 1 else np.array([])

        # build data for the mixed bar plot: observed + all permuted
        vals, nets, is_orig = [], [], []

        # originals first
        vals.extend(obs_vals.tolist())
        nets.extend(NET_list)
        is_orig.extend([True] * n_networks)

        # then permutations
        if n_perms > 1:
            nets.extend(np.tile(NET_list, n_perms - 1).tolist())
            vals.extend(carr[:, 1:, j].reshape(-1).tolist())
            is_orig.extend([False] * (n_networks * (n_perms - 1)))

        vals = np.asarray(vals)
        nets = np.asarray(nets)
        is_orig = np.asarray(is_orig, dtype=bool)

        # sort by bar height once
        order = np.argsort(vals)[::-1]
        vals = vals[order]
        nets = nets[order]
        is_orig = is_orig[order]

        # percentiles from permuted only (if available) — used on LEFT panel
        have_perms = perm_vals.size > 0
        if have_perms:
            p95 = np.percentile(perm_vals, 95)
            p99 = np.percentile(perm_vals, 99)
        else:
            p95 = p99 = np.nan

        # ---------- figure: single null-distribution panel ----------
        from matplotlib.lines import Line2D  # Patch no longer needed

        fig, ax_hist = plt.subplots(figsize=(12, 5))

        ax_hist.set_title(f"Null-Distribution – {feature_name}")
        ax_hist.set_xlabel("Correlation")

        if have_perms:
            # KDE curve (no rectangles)
            try:
                from scipy.stats import gaussian_kde

                xs = np.linspace(np.min(perm_vals), np.max(perm_vals), 400)
                kde = gaussian_kde(perm_vals)
                ax_hist.plot(xs, kde(xs), color='k', linewidth=2, label='Shuffled density')
                ax_hist.set_ylabel("Density")
            except Exception:
                # Fallback: line through histogram counts (still no bars)
                counts, edges = np.histogram(perm_vals, bins=20)
                centers = 0.5 * (edges[:-1] + edges[1:])
                ax_hist.plot(centers, counts, color='k', linewidth=2, label='Shuffled (line)')
                ax_hist.set_ylabel("Count")

            # observed (blue) at the mean of per-net observed values
            corr_obs = float(np.mean(obs_vals))
            ax_hist.axvline(corr_obs, color='tab:blue', linestyle='-', linewidth=3,
                            label=f'Observed = {corr_obs:.3f}')

            # red dashed 95th percentile
            p95 = np.percentile(perm_vals, 95)
            ax_hist.axvline(p95, color='tab:red', linestyle='--', linewidth=2,
                            label=f'95th = {p95:.3f}')

            # overlay dashed colored lines for each network's observed value
            for n, v in zip(NET_list, obs_vals):
                ax_hist.axvline(v, linestyle='--', linewidth=1.5, color=net_colors[n], alpha=0.9)

            # legends: keep only line legends (no square patches)
            leg1 = ax_hist.legend(frameon=False, loc="upper left")
            ax_hist.add_artist(leg1)

            net_line_handles = [Line2D([0], [0], color=net_colors[n], linestyle='--', label=n) for n in NET_list]
            ax_hist.legend(handles=net_line_handles, title="Observed by network",
                           bbox_to_anchor=(1.02, 1), loc="upper left", frameon=False)

        else:
            ax_hist.text(0.5, 0.5, "No permutations\n(n_perms == 1)",
                         ha="center", va="center", transform=ax_hist.transAxes)
            ax_hist.set_xticks([]);
            ax_hist.set_yticks([])

        plt.tight_layout()
        plt.show()


elif test != "300" and test != "rest":
    # Classic grouped bar chart
    x = np.arange(n_networks)
    bar_width = 0.8 / n_features
    feat_colors = [cmap(i / n_features) for i in range(n_features)]

    plt.figure(figsize=(12, 6))
    for j in range(n_features):
        plt.bar(
            x + j * bar_width,
            cor_array[:, j],
            width=bar_width,
            label=feature_names[j],
            color=feat_colors[j]
        )
    if test == "movies" and directory != r"D:\Final Project\Predicting Human Brain States with Transformer\Gal&Yuval code\Processed_Matrices":
        # cor_array currently holds rows for every (NET, movie) pair, columns = features
        arr_all = np.array(cor_array)  # shape: [n_networks * n_movies, n_features]

        n_features = len(feature_names)
        n_movies = 14 if directory == r"F:\HCP_movie_base_norm" else 15
        n_networks = len(NET_list)

        expected = n_networks * n_movies
        if arr_all.shape[0] != expected:
            raise ValueError(
                f"Expected {expected} rows in cor_array, but got {arr_all.shape[0]}. "
                "Check that you didn't skip any network/movie during training."
            )

        # 3D view for convenience: [net × movie × feature]
        arr3 = arr_all.reshape((n_networks, n_movies, n_features))

        # 2D view (your original plotting trick): [net × (movie * feature)]
        corr2d = arr3.reshape((n_networks, n_movies * n_features))

        movie_nums = np.arange(n_movies)
        bar_width = 0.8 / n_features
        feat_colors = [cmap(i / n_features) for i in range(n_features)]

        # ---- Existing per-NET plots across movies (unchanged logic, just using corr2d) ----
        for net_idx in range(n_networks):
            plt.figure(figsize=(12, 6))

            for feat_idx in range(n_features):
                # each feature appears every n_features columns
                bar_vals = corr2d[net_idx, feat_idx::n_features]
                plt.bar(
                    movie_nums + feat_idx * bar_width,
                    bar_vals,
                    width=bar_width,
                    label=feature_names[feat_idx],
                    color=feat_colors[feat_idx]
                )

            plt.xticks(movie_nums + bar_width * (n_features - 1) / 2,
                       [f"Movie {i}" for i in (movie_nums + (1 if directory == r'F:\HCP_movie_base_norm' else 0))],
                       rotation=45)
            plt.xlabel("Movie Number")
            plt.ylabel("Correlation")
            plt.title(f"Diagonal Correlation Values – Network: {NET_list[net_idx]}")
            plt.grid(True, axis="y", linestyle="--", alpha=0.6)
            plt.ylim(-0.6, 0.6)
            plt.yticks(np.arange(-0.6, 0.61, 0.1))
            plt.legend(title="Feature", bbox_to_anchor=(1.05, 1), loc="upper left")
            plt.tight_layout()
            plt.show()

        # ---- NEW: Best movie's feature vector for each NET ----
        # Score each movie by mean absolute correlation across features, pick the argmax.
        best_movie_idx = []
        best_scores = []
        for net_idx in range(n_networks):
            scores = np.mean(np.abs(arr3[net_idx, :, :]), axis=1)  # [n_movies]
            bm = int(np.argmax(scores))
            best_movie_idx.append(bm)
            best_scores.append(float(scores[bm]))

        # Plot best-movie features for each NET in a grid of subplots
        n_cols = min(3, n_networks)
        n_rows = int(np.ceil(n_networks / n_cols))
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows), sharey=True)
        axes = np.atleast_1d(axes).reshape(n_rows, n_cols)

        for net_idx, ax in enumerate(axes.flat[:n_networks]):
            bm = best_movie_idx[net_idx]
            vals = arr3[net_idx, bm, :]  # feature vector for best movie
            ax.bar(np.arange(n_features), vals, color=feat_colors)
            ax.set_xticks(np.arange(n_features))
            ax.set_xticklabels(feature_names, rotation=45, ha='right', fontsize=8)
            label_movie_num = bm + (1 if directory == r"F:\HCP_movie_base_norm" else 0)
            ax.set_title(f"{NET_list[net_idx]} – Best: Movie {label_movie_num} (mean|r|={best_scores[net_idx]:.2f})")
            ax.set_ylim(-0.6, 0.6)
            ax.grid(True, axis="y", linestyle="--", alpha=0.6)

        # Hide any empty subplots
        for ax in axes.flat[n_networks:]:
            ax.set_visible(False)

        fig.suptitle("Best Movie Feature Correlations per Network", y=1.02)
        fig.tight_layout()
        plt.show()


    else:
        plt.xticks(x + bar_width * (n_features - 1) / 2, NET_list, rotation=45)
        plt.xlabel('Network')
        plt.ylabel('Correlation')
        plt.title('Diagonal Correlation Values per Network')
        plt.legend(title='Feature', bbox_to_anchor=(1.05, 1), loc='upper left')
        plt.grid(True)
        plt.yticks(np.arange(-0.6, 0.7, 0.1))  # y-axis ticks from -0.6 to 0.6 in steps of 0.1
        plt.tight_layout()
        plt.show()

else:
    # Plot one sorted bar graph per feature + "best file per group" barplot (colors = color_map)
    for j in range(n_features):
        feature_name = feature_names[j]
        values = cor_array[:, j]  # aligned with NET_list

        # ---------- 1) Sorted full list (unchanged, but legend colors from color_map) ----------
        sorted_indices = np.argsort(values)[::-1]
        sorted_values = values[sorted_indices]
        sorted_nets = [NET_list[idx] for idx in sorted_indices]

        # Colors for bars: group color from color_map via middle token *_NET_*
        sorted_groups = [name.split('_')[1] if '_' in name else name for name in sorted_nets]
        sorted_colors = [color_map[g] for g in sorted_groups]

        print(f"\nTop 5 networks for feature: {feature_name}")
        for k in range(100):
            print(f"{k + 1}. {sorted_nets[k]} — Correlation: {sorted_values[k]:.4f}")

        plt.figure(figsize=(max(14, len(sorted_nets) * 0.4), 8))
        plt.bar(range(n_networks), sorted_values, color=sorted_colors)

        plt.xticks(
            ticks=range(n_networks),
            labels=sorted_nets,               # full filename labels
            rotation=90,
            fontsize=max(8, int(300 / n_networks))
        )
        plt.ylabel('Correlation')
        plt.title(f'Correlation per Network – {feature_name}')
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        from matplotlib.patches import Patch
        # Legend by group, colors from color_map (ensures exact match)
        groups_in_plot = []
        for g in sorted_groups:
            if g not in groups_in_plot:
                groups_in_plot.append(g)
        legend_patches = [Patch(color=color_map[g], label=g) for g in groups_in_plot]
        plt.legend(handles=legend_patches, title='Network Group', bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.35)  # room for long filenames
        plt.show()

        # ---------- 2) NEW: Best performing file per network group (full filenames as labels) ----------
        file_groups = [nm.split('_')[1] if '_' in nm else nm for nm in NET_list]

        canonical = ['Cont', 'Default', 'DorsAttn', 'Limbic', 'SalVentAttn', 'SomMot', 'Vis']
        groups_present = [g for g in canonical if g in file_groups]  # keep stable order

        best_files, best_vals, best_colors = [], [], []
        for g in groups_present:
            idxs = [idx for idx, gg in enumerate(file_groups) if gg == g]
            best_idx = idxs[int(np.argmax(values[idxs]))]
            best_files.append(NET_list[best_idx])          # full filename (visible under bar)
            best_vals.append(float(values[best_idx]))
            best_colors.append(color_map[g])               # <-- color from color_map to match previous plot

        plt.figure(figsize=(max(12, 2.0 * len(best_files)), 6))
        x = np.arange(len(best_files))
        plt.bar(x, best_vals, color=best_colors)

        plt.xticks(x, best_files, rotation=90, fontsize=11)  # full filenames; make sure readable
        plt.ylabel('Correlation')
        plt.title(f'Best Performing File per Network Group – {feature_name}')
        plt.grid(axis='y', linestyle='--', alpha=0.5)

        # Legend uses the exact same colors as above (from color_map)
        legend_patches = [Patch(color=color_map[g], label=g) for g in groups_present]
        plt.legend(handles=legend_patches, title='Network Group', bbox_to_anchor=(1.02, 1), loc='upper left')

        plt.tight_layout()
        plt.subplots_adjust(bottom=0.4)  # extra room for long labels
        plt.show()
