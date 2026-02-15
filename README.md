# Federated MF with Differential Privacy and MIA

Pipeline:

1. Train MF model
2. Evaluate RMSE
3. Extract scores for MIA
4. Build attack dataset
5. Train attack classifier

Everything is script-based and reproducible via seeds.


{'model': 'FedDP_MF', 'k': 64, 'lr': 0.05, 'local_epochs': 16, 'rounds': 80, 'sample_rate': 0.1, 'batch_size': 16, 'reg': 0.01, 'clip_norm': 1.0, 'noise_multiplier': 0.0, 'seed': 0, 'test_frac': 0.2, 'delta': 1e-06}