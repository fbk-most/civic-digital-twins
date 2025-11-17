(t_max, e_max) = (10000, 10000)
(t_sample, e_sample) = (100, 100)
target_presence_samples = 200
ensemble_size = 20

early_stopping_params = {
    "batch_size": 20,
    "min_batch_iterations": 1,
    "max_batch_iterations": 20,
    "evaluate_every_n_batches": 1,
    "confidence_threshold": 0.8,
    "stability_tolerance": 1e-3,
}
