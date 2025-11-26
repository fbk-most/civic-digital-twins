from pathlib import Path
import pickle

use_dask = True
save = True

params = {
    "t_max": 10000,
    "e_max": 10000,
    "t_sample": 100,
    "e_sample": 100,
    "target_presence_samples": 200,
    "ensemble_size": 20,
    "early_stopping_params": {
        "batch_size": 20,
        "min_batch_iterations": 1,
        "max_batch_iterations": 20,
        "evaluate_every_n_batches": 1,
        "confidence_threshold": 0.8,
        "stability_tolerance": 1e-3,
    },
    "early_stopping": False,
}

if __name__ == "__main__":
    scenarios = {
        "Base": {},
        "GoodWeather": {"CV_weather": ["good", "unsettled"]},
        "BadWeather": {"CV_weather": ["bad"]},
    }

    from civic_digital_twins.dt_model.simulation.utils import (
        compute_scenario_worker,
        plot_scenario,
    )

    if use_dask:
        from dask import delayed, compute

        delayed_results = [
            delayed(compute_scenario_worker)(config, params)
            for config in scenarios.values()
        ]

        scenario_results = compute(*delayed_results)  # Parallel execution
        for name, result in zip(scenarios.keys(), scenario_results):
            scenario_name, data = result
            print(f"Scenario {name} completed: {scenario_name}")
            if save:
                # Build output path and save
                OUTPUT_DIR = Path("scenario_results")
                OUTPUT_DIR.mkdir(exist_ok=True)
                outfile = OUTPUT_DIR / f"{name}_{scenario_name}.pkl"
                with open(outfile, "wb") as f:
                    pickle.dump(data, f)

                plot_scenario(data, filename=OUTPUT_DIR / f"{name}_plot.png")

    else:
        for name, config in scenarios.items():

            scenario_name, data = compute_scenario_worker(config, params)
            if save:
                # Build output path and save
                OUTPUT_DIR = Path("scenario_results")
                OUTPUT_DIR.mkdir(exist_ok=True)
                outfile = OUTPUT_DIR / f"{name}_{scenario_name}.pkl"
                with open(outfile, "wb") as f:
                    pickle.dump(data, f)

                plot_scenario(data, filename=OUTPUT_DIR / f"{name}_plot.png")
