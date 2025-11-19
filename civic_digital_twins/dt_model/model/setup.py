params = {
    "t_max": 10000,
    "e_max": 10000,
    "t_sample": 100,
    "e_sample": 100,
    "target_presence_samples": 200,
    "ensemble_size": 20,
    "batch_size": 20,
    "early_stopping_params": {
        "batch_size": 20,
        "min_batch_iterations": 1,
        "max_batch_iterations": 20,
        "evaluate_every_n_batches": 1,
        "confidence_threshold": 0.8,
        "stability_tolerance": 1e-3
    },
    "early_stopping": True,
    "dag_mode": False # can be true only if eary stopping is false
}