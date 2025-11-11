import os
import numpy as np
import pickle
from pathlib import Path
import time
import concurrent.futures
import threading
import queue
import psutil
import signal
import hashlib

import matplotlib.pyplot as plt
from matplotlib.cm import ScalarMappable
from matplotlib.colors import Normalize

from civic_digital_twins.dt_model import InstantiatedModel
from civic_digital_twins.dt_model.internal.sympyke import Symbol
from civic_digital_twins.dt_model import Ensemble, Evaluation
from civic_digital_twins.dt_model.reference_models.molveno.overtourism import (
    I_P_excursionists_reduction_factor,
    I_P_excursionists_saturation_level,
    I_P_tourists_reduction_factor,
    I_P_tourists_saturation_level,
    PV_excursionists,
    PV_tourists,
    M_Base,
    CV_weather,
)

OUTPUT_DIR = Path("scenario_results")


class ScenarioManager:
    def __init__(self, max_workers=4, cpu_threshold=0.8):
        self.max_workers = max_workers or psutil.cpu_count(logical=False)
        self.cpu_threshold = cpu_threshold
        self.executor = concurrent.futures.ProcessPoolExecutor(
            max_workers=self.max_workers,
            mp_context=__import__("multiprocessing").get_context("spawn"),
        )

        self.task_queue = queue.Queue()
        self.running = True
        self._lock = threading.Lock()
        self._active = {}  # name -> (future, pid)
        threading.Thread(target=self._dispatcher, daemon=True).start()

    def _dispatcher(self):
        while self.running:
            cpu_load = psutil.cpu_percent(interval=0.2)
            if cpu_load > self.cpu_threshold * 100:
                time.sleep(0.5)
                continue

            if len(self._active) < self.max_workers and not self.task_queue.empty():
                name, params = self.task_queue.get()
                try:
                    future = self.executor.submit(compute_scenario_worker, params)
                    with self._lock:
                        self._active[name] = (future, None)
                    future.add_done_callback(lambda f, n=name: self._on_done(n, f))
                finally:
                    self.task_queue.task_done()
            else:
                time.sleep(0.1)

    def _on_done(self, name, future):
        with self._lock:
            self._active.pop(name, None)
        try:
            _ = future.result()
            print(f"{name} finished successfully")
        except Exception as e:
            print(f"{name} failed: {e}")

    def submit(self, name, params=None):
        """Queue a new scenario computation."""
        self.task_queue.put((name, params or {}))

    def cancel(self, name):
        """Cancel a queued or running scenario."""
        with self._lock:
            if name in self._active:
                future, pid = self._active[name]
                if pid:
                    try:
                        os.kill(pid, signal.SIGTERM)
                        print(f"Terminated process for {name}")
                    except Exception as e:
                        print(f"Could not terminate {name}: {e}")
                future.cancel()
                self._active.pop(name, None)
                return True

        # If not active, try to remove from queue
        removed = []
        while not self.task_queue.empty():
            n, params = self.task_queue.get()
            if n != name:
                removed.append((n, params))
        for job in removed:
            self.task_queue.put(job)
        if removed:
            print(f"Removed '{name}' from queue.")
            return True

        return False

    def shutdown(self):
        self.running = False
        self.executor.shutdown(wait=False)
        print("Scenario manager stopped.")


def presence_transformation(presence, reduction_factor, saturation_level, sharpness=3):
    tmp = presence * reduction_factor
    return (
        tmp
        * saturation_level
        / ((tmp**sharpness + saturation_level**sharpness) ** (1 / sharpness))
    )


def compute_scenario(model, scenario_config):
    """Compute all data for a given scenario"""

    # static data, might parameters later
    t_max, e_max = 10000, 10000
    t_sample, e_sample = 800, 800
    target_presence_samples = 200
    ensemble_size = 20

    ensemble = Ensemble(model, scenario_config, cv_ensemble_size=ensemble_size)
    scenario_hash = ensemble.compute_hash(
        [t_max, e_max, t_sample, e_sample, target_presence_samples], "molveno"
    )
    evaluation = Evaluation(model, ensemble)

    tt = np.linspace(0, t_max, t_sample + 1)
    ee = np.linspace(0, e_max, e_sample + 1)
    zz = evaluation.evaluate_grid({PV_tourists: tt, PV_excursionists: ee})

    sample_tourists = [
        presence_transformation(
            presence,
            evaluation.get_index_mean_value(I_P_tourists_reduction_factor),
            evaluation.get_index_mean_value(I_P_tourists_saturation_level),
        )
        for c in ensemble
        for presence in PV_tourists.sample(
            cvs=c[1], nr=max(1, round(c[0] * target_presence_samples))
        )
    ]

    sample_excursionists = [
        presence_transformation(
            presence,
            evaluation.get_index_mean_value(I_P_excursionists_reduction_factor),
            evaluation.get_index_mean_value(I_P_excursionists_saturation_level),
        )
        for c in ensemble
        for presence in PV_excursionists.sample(
            cvs=c[1], nr=max(1, round(c[0] * target_presence_samples))
        )
    ]

    return {
        "evaluation": evaluation,
        "zz": zz,
        "sample_tourists": sample_tourists,
        "sample_excursionists": sample_excursionists,
        "t_max": t_max,
        "t_sample": t_sample,
        "e_max": e_max,
        "e_sample": e_sample,
    }, scenario_hash


def compute_scenario_worker(scenario_config: dict):
    """
    Compute one scenario and save the results to disk.

    Args:
        scenario_name: name of the scenario ("Base", "GoodWeather", etc.)
        params: optional dict with custom parameters (future-proof)
    Returns:
        Path to the saved results file.
    """
    start = time.time()
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Each worker builds its own model
    IM_Base = InstantiatedModel(M_Base, values=scenario_config)

    # Compute scenario data
    result, scenario_name = compute_scenario(IM_Base, scenario_config)

    # Build output path and save
    outfile = OUTPUT_DIR / f"{scenario_name}.pkl"
    with open(outfile, "wb") as f:
        pickle.dump(result, f)

    print(f"Scenario '{scenario_name}' completed in {time.time() - start:.2f}s")


def scenario_name_from_config(config: dict) -> str:
    config_items = []
    for k, v in config.items():
        v_str = "-".join(str(x) for x in v) if isinstance(v, (list, tuple)) else str(v)
        config_items.append(f"{getattr(k, 'name', str(k))}_{v_str}")
    config_str = "_".join(config_items)
    # Add a short hash suffix to avoid collisions
    hash_code = hashlib.sha1(config_str.encode()).hexdigest()[:6]
    return f"{config_str}_{hash_code}" if config_str else f"default_{hash_code}"


def plot_scenario(data, filename=None):
    """Plot a single scenario using precomputed data."""

    # Compute relevant parts
    tt = np.linspace(0, data["t_max"], data["t_sample"] + 1)
    ee = np.linspace(0, data["e_max"], data["e_sample"] + 1)
    xx, yy = np.meshgrid(tt, ee)
    evaluation = data["evaluation"]

    area = evaluation.compute_sustainable_area()
    i, c = evaluation.compute_sustainability_index_with_ci(
        list(zip(data["sample_tourists"], data["sample_excursionists"])), confidence=0.8
    )
    sust_indexes = evaluation.compute_sustainability_index_with_ci_per_constraint(
        list(zip(data["sample_tourists"], data["sample_excursionists"])), confidence=0.8
    )
    critical = min(sust_indexes, key=lambda k: sust_indexes[k][0])
    modals = evaluation.compute_modal_line_per_constraint()

    fig, ax = plt.subplots(figsize=(8, 6))
    pcm = ax.pcolormesh(xx, yy, data["zz"], cmap="coolwarm_r", vmin=0.0, vmax=1.0)

    for modal in modals.values():
        ax.plot(*modal, color="black", linewidth=2)

    ax.scatter(
        data["sample_excursionists"],
        data["sample_tourists"],
        color="gainsboro",
        edgecolors="black",
    )

    critical_mean, critical_ci = sust_indexes[critical]
    ax.set_title(
        f"Area = {area / 10e6:.2f} kp$^2$ - "
        f"Sustainability = {i * 100:.2f}% ± {c * 100:.2f}%\n"
        f"Critical = {critical.capacity.name}"
        f" ({critical_mean * 100:.2f}% ± {critical_ci * 100:.2f}%)",
        fontsize=12,
    )
    ax.set_xlim(0, data["t_max"])
    ax.set_ylim(0, data["e_max"])
    fig.colorbar(ScalarMappable(Normalize(0, 1), cmap="coolwarm_r"), ax=ax)
    ax.set_xlabel("Tourists")
    ax.set_ylabel("Excursionists")

    if filename:
        plt.savefig(filename, bbox_inches="tight")
    else:
        plt.show()


if __name__ == "__main__":
    ##############################
    #    COMPUTATIONAL PART      #
    ##############################

    start = time.time()
    manager = ScenarioManager(max_workers=4)

    manager.submit("Base", {})
    manager.submit("GoodWeather", {CV_weather: [Symbol("good"), Symbol("unsettled")]})
    manager.submit("BadWeather", {CV_weather: [Symbol("bad")]})

    # Wait for all to finish
    while manager._active or not manager.task_queue.empty():
        time.sleep(1)

    manager.shutdown()
    elapsed = time.time() - start
    print(
        f"Computation step: {elapsed:.2f}"
        if elapsed >= 0
        else f"Computation step: <0 ({elapsed:.2f})"
    )

    ##############################
    #         PLOTTING           #
    ##############################
    start = time.time()
    result_dir = Path("scenario_results")

    for file in result_dir.glob("*.pkl"):
        name = file.stem
        with open(file, "rb") as f:
            data = pickle.load(f)
        plot_scenario(data, filename=f"{name.lower()}_plot.png")

    elapsed = time.time() - start
    print(
        f"Plotting: {elapsed:.2f}"
        if elapsed >= 0
        else f"Plotting step: <0 ({elapsed:.2f})"
    )
