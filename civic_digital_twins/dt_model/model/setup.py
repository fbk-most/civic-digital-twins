params = {
    "t_max": 10000,
    "e_max": 10000,
    "t_sample": 100,
    "e_sample": 100,
    "target_presence_samples": 200,
    "ensemble_size": 20,
<<<<<<< HEAD
=======
    "batch_size": 20,
>>>>>>> 9b5348b455512a044958c67e9c9f8c9517991b05
    "early_stopping_params": {
        "batch_size": 20,
        "min_batch_iterations": 1,
        "max_batch_iterations": 20,
        "evaluate_every_n_batches": 1,
        "confidence_threshold": 0.8,
<<<<<<< HEAD
        "stability_tolerance": 1e-3,
    },
    "early_stopping": True,
    "use_dask": False,
}
=======
        "stability_tolerance": 1e-3
    },
    "early_stopping": True,
    "dag_mode": False # can be true only if eary stopping is false
}
>>>>>>> 9b5348b455512a044958c67e9c9f8c9517991b05
