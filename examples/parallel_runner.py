import digitalhub as dh
from digitalhub_runtime_python import handler
from civic_digital_twins.dt_model.simulation.utils import (
    compute_scenario_worker,
)
import pickle


@handler()
def main(project, name=None, config={}, in_params={}):
    # download as local file
    scenario_name, result = compute_scenario_worker(config, in_params)
    outfile = f"{name}_{scenario_name}.pkl"

    with open(outfile, "wb") as f:
        pickle.dump(result, f)

    project.log_artifact(kind="artifact", name=outfile, source=outfile)
