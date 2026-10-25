# BT4012-kaggle-comp

Bitcoin transaction classification using the Elliptic competition data. The final pipeline combines a supervised branch with an illicit-versus-rest (IR) branch, then applies a single incoming-neighbour risk correction.

## Setup and data

Install the dependencies listed in `requirements.txt` into the Python environment used to run the notebooks:

```bash
python -m pip install -r requirements.txt
```

The notebooks were developed in Google Colab. For local execution, remove the Google Drive import/mount calls, set `ROOT_DIR = "."`, and run the notebooks with the repository root as the working directory. Open them in Jupyter or a notebook-capable editor using the environment above.

Download the competition data separately and place these files in the repository:

- `base_datasets/train.csv`
- `base_datasets/test.csv`
- `base_datasets/txs_edgelist.csv`
- `base_datasets/sample_submission.csv`

Create the output directories `transformed_datasets/`, `base_models_cache/` and `submissions/` before running, or create them in the setup cells. The datasets and generated model caches do not need to be committed to GitHub.

For Colab execution, use a shared project folder accessible from every notebook, mount Drive if needed, and set `ROOT_DIR` to that folder. Keep the same project folder throughout the workflow so that later notebooks can read earlier outputs.

## Notebook execution order

1. **`BT4012_competition_EDA.ipynb`** - explore the data, build topology features and export `transformed_datasets/train_topo.csv` and `test_topo.csv`.
2. **`BT4012_competition_baseline_modelling.ipynb`** - fit the supervised models on known-label transactions and save their models, OOF predictions and test predictions in `base_models_cache/`.
3. **`BT4012_competition_baseline_modelling_unlabelled_ir.ipynb`** - fit XGBoost, LightGBM and CatBoost with known illicit transactions assigned target 1 and all other training transactions assigned target 0. Save `xgb_ir.joblib`, `lgbm_ir.joblib` and `catboost_ir.joblib`.
4. **`BT4012_competition_final_ensemble_modelling.ipynb`** - load the saved predictions, apply the chosen model/branch weights and neighbour correction, and export a submission CSV.

`BT4012_competition_GNN_exploration.ipynb` is an optional experiment and is not needed to reproduce the selected ensemble submissions. It additionally requires `torch` and `torch-geometric`.

## Grid search and saved parameters

Grid searches can take a long time. In the supervised baseline notebook, skip the GSCV cells and run the parameter-tuning cells containing `SimpleNamespace` with the saved best hyperparameters. Continue through temporal validation and prediction/persistence cells to regenerate the caches. CatBoost's seed-averaging and training cells must still be run.

In the IR notebook, skip a search only if a separate saved-parameter cell defines the corresponding model. If the model is defined at the end of its GSCV cell, that cell must be run or replaced with the saved configuration first. The selected raw-feature IR configurations are:

| Model | Learning rate | Trees / iterations | Tree complexity |
|---|---:|---:|---|
| XGBoost IR | 0.05 | 1000 | `max_depth=4` |
| LightGBM IR | 0.05 | 1000 | `num_leaves=31` |
| CatBoost IR | 0.05 | 500 | `depth=4` |

Keep the other parameters from each model-definition cell. IR models use seed 42. The supervised CatBoost ensemble uses the seed configuration defined in the supervised notebook.

The selected submissions use only the original 165 `feat_*` columns as model inputs. Engineered topology columns are retained in the exported data for exploration but are not selected for these submissions. Neighbour risk uses the edge list separately.

## Reproduce the two highest recorded submissions

Generate the base caches once using the workflow above. In the final ensemble notebook, run all cells for the first submission. For the second, change only `CONFIG` and rerun from **Experiment Configuration** to the end. No base-model retraining is required between these two configurations.

### Submission 1

Recorded public ROC-AUC: **0.96019**.

```python
CONFIG = {
    "sup_weights": {"xgb": 0.75, "catboost": 0.25},
    "ir_weights": {"xgb_ir": 0.60, "lgbm_ir": 0.10, "catboost_ir": 0.30},
    "ir_blend_weight": 0.80,
    "risk_weight": 0.30,
}
```

Output:

`submissions/sup_xgb75_catboost25__ir_xgb_ir60_lgbm_ir10_catboost_ir30__mix80_risk30_submission.csv`

### Submission 2

Recorded public ROC-AUC: **0.95991**.

```python
CONFIG = {
    "sup_weights": {"xgb": 0.75, "catboost": 0.25},
    "ir_weights": {"xgb_ir": 0.55, "lgbm_ir": 0.10, "catboost_ir": 0.35},
    "ir_blend_weight": 0.80,
    "risk_weight": 0.30,
}
```

Output:

`submissions/sup_xgb75_catboost25__ir_xgb_ir55_lgbm_ir10_catboost_ir35__mix80_risk30_submission.csv`

Model probabilities are averaged within each branch. Branch predictions are combined in log-odds space, then corrected once using available incoming-neighbour predictions. The risk weight is reduced according to neighbour prediction coverage. Adjusted predictions are not propagated again.

Public scores above are recorded competition results. Reproducing the same predictions requires the same input data, feature columns, model settings, seed setup and compatible package versions. Regenerating a cache overwrites its previous version, so regenerate all selected caches from the intended configurations before generating the final submissions.
