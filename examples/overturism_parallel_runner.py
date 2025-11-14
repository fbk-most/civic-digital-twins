from pathlib import Path
import pickle

EARLY_STOPPING = False
if EARLY_STOPPING:
    EARLY_STOPPING_PARAMS = {
        "batch_size": 20,
        "min_iterations": 1,
        "max_batches": 20,
        "confidence_threshold": 0.8,
        "stability_tolerance": 1e-3,
    }
save = True

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

    for name, config in scenarios.items():

        scenario_name, data = compute_scenario_worker(config, EARLY_STOPPING)
        if save:
            # Build output path and save
            OUTPUT_DIR = Path("scenario_results")
            OUTPUT_DIR.mkdir(exist_ok=True)
            outfile = OUTPUT_DIR / f"{name}_{scenario_name}.pkl"
            with open(outfile, "wb") as f:
                pickle.dump(data, f)

        plot_scenario(data, filename=OUTPUT_DIR / f"{name}_plot.png")
