from civic_digital_twins.dt_model.simulation.overturism_parallel import (
    compute_scenario_worker,
    plot_scenario,
)

EARLY_STOPPING = False
if EARLY_STOPPING:
    EARLY_STOPPING_PARAMS = {
        "batch_size": 20,
        "min_iterations": 1,
        "max_batches": 20,
        "confidence_threshold": 0.8,
        "stability_tolerance": 1e-3,
    }
PARALLEL_EXECUTION = False


if __name__ == "__main__":
    scenarios = {
        "Base": {},
        "GoodWeather": {"CV_weather": ["good", "unsettled"]},
        "BadWeather": {"CV_weather": ["bad"]},
    }

    for name, config in scenarios.items():

        outfile, data = compute_scenario_worker(
            config, PARALLEL_EXECUTION, EARLY_STOPPING
        )
        plot_scenario(data, filename=f"{name.lower()}_plot.png")
