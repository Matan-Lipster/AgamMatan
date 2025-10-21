# fMRI Transformer and PLS Modeling

This repository contains several versions of our fMRI analysis workflow.  
The project is organized around three main components:  

- **main** – handles the main execution logic  
- **preprocess** – prepares and structures the fMRI data  
- **model** – defines and trains the predictive model  

Within the repository you will also find variations of these files.  
These were created as part of our experimentation with different model types and datasets.  

In addition, there is a small utility file named **`delete.py`**, used for printing and inspecting specific fMRI `.pkl` files.

---

## Overview of Versions

Throughout development, we tested three major configurations of the code:

1. **Original Transformer and Dataset**  
   Our baseline implementation using our own dataset and Transformer model.

2. **Collaborators’ Transformer and Dataset (`gy_` files)**  
   Versions based on a model and dataset provided by our colleagues.  
   These files are prefixed with `gy_`.

3. **PLS Model Experiments (`pls_` files)**  
   Versions that replace the Transformer with a Partial Least Squares (PLS) model, tested with multiple datasets and configuration options.  
   These files are prefixed with `pls_`.

---

## Manual

### 1. Running the Transformer Workflow

1. Open **`main_try.py`**.  
   Inside, you can modify all Transformer and dataset parameters.  

   The key parameters that control what part of the brain is analyzed are:  
   - **`NET_list`** – defines the sub-networks to include.  
   - **`NET_indexes`** – specifies the regions of interest (ROIs) within those networks.  
   - **`H_list`** – selects which hemispheres to include (left, right, or both).  

   You can select multiple values for each parameter.  

2. Run **`main_try.py`** with your chosen settings.  
3. When prompted, press **1** to select the option that outputs the unique features related to fluid intelligence.

---

### 2. Using the Collaborators’ Transformer (`gy_` Files)

1. Open **`gy_main.py`** and modify the same key parameters:  
   `NET_list`, `NET_indexes`, and `H_list`.  

2. Run **`gy_main.py`** with your selected settings.  
3. When prompted, press **1** to extract the fluid intelligence features.  

   *Note:* This workflow combines all selected ROIs into one matrix and averages the data accordingly.

---

### 3. Using the PLS Model (`pls_` Files)

1. Open **`pls_main.py`**.  
   This file provides several options for both the **features** and the **data** to be tested.  

   **Feature options:**  
   - `fluid intelligence`  
   - `personality`  
   - `fluid intelligence 2` (two out of three intelligence features)  
   - `all` (includes all features)

   **Data options:**  
   - `1` – run on specific ROIs  (by putting them in NET_list)  
   - `2` – run on sub-networks   (by putting them in NET_list)  
   - `3` – run on full networks  (by putting them in NET_list)  
   - `300` – run on all 300 regions  
   - `rest` – resting-state data, all regions  
   - `movies` – per-movie data  
   - `all` – all networks and all features  

2. The dataset path is defined in the `directory` parameter.  
   The default is:  
   ```python
   r"F:\HCP_DATA"
   ```  
   You can uncomment or modify other directory paths inside the file to use alternative datasets.

3. Any option larger than an ROI (for example, networks, subnetworks, or movies) will automatically apply averaging across matrices.  
   Alternatively, you can apply **PCA** or **MEAN** instead.  
   To enable 100 random permutations, set:  
   ```python
   MUTATE = True
   ```

---

## Notes

- All workflows expect the fMRI data to be stored in `.pkl` format.  
- Averaging is automatically applied when aggregating data from larger structures.  
- The `delete.py` script is used only for inspecting specific fMRI pickle files.

---

### Example

```bash
python main_try.py
python gy_main.py
python pls_main.py
```

Each script can be run independently with its respective parameters.

---

## Summary

This repository includes three parallel pipelines exploring different approaches to modeling fMRI data:

- **Baseline Transformer approach**  
- **Collaborator Transformer adaptation (`gy_`)**  
- **PLS-based modeling (`pls_`)**

All versions share the same general structure but differ in their data sources, model types, and preprocessing methods.
