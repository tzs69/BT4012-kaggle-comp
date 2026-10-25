# BT4012-kaggle-competition
For reproduction of submission results, run the notebooks in this order:
1) BT4012_competition_EDA.ipynb
2) BT4012_competition_baseline_modelling.ipynb
3) BT4012_competition_baseline_modelling_unlabelled_ir.ipynb
4) BT4012_competition_final_ensemble_modelling.ipynb

For the baseline modelling notebooks, grid searches can take a long time. To reproduce the selected models without repeating the searches, skip the GSCV cells and run the parameter-tuning cells containing SimpleNamespace with the saved best hyperparameters, usually located immediately after the GSCV cells.
