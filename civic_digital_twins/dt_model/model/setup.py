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

EARLY_STOPPING = True
DAG_MODE = False
if EARLY_STOPPING:
    # TODO [federicalago] find better way to ensure this constraint
    DAG_MODE = False

# Compute evaluation rules
# TODO [federicalago] this can be improved with an ML/statistical analysis of elaboration times

ENSEMBLE_BATCH_THRESHOLD = 50  # adjustable
USE_ENSEMBLE_BATCHING = not EARLY_STOPPING and ensemble_size >= ENSEMBLE_BATCH_THRESHOLD

# grid_points = (t_sample + 1) * (e_sample + 1)
# GRID_BIG_THRESHOLD = 50000
# COMPRESS_ARTIFACTS = not EARLY_STOPPING and (
#     (grid_points > GRID_BIG_THRESHOLD) or ensemble_size >= ENSEMBLE_BATCH_THRESHOLD
# )

# use_local_threading = early_stopping and ensemble_size > 1

# simple_merge = (ensemble_size <= 5)
