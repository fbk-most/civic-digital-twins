import json
import os

from hera.workflows import Workflow, DAG, script, Resources

DEFAULT_IMAGE = os.environ.get("DH_WORKFLOW_IMAGE", "python:3.12-slim")


# -----------------------------
# SCRIPT: compute a sub-batch
# -----------------------------
@script(
    image=DEFAULT_IMAGE,
    resources=Resources(memory_request="2Gi"),
)
def compute_subensemble(sub_json: str, scenario_json: str, out_file: str):
    """
    sub_json: JSON list of sub-ensemble items
    scenario_json: JSON scenario config
    out_file: path to write pickle with field/conf
    """
    import json, pickle, os
    from civic_digital_twins.dt_model.model.instantiated_model import InstantiatedModel

    from civic_digital_twins.dt_model.engine import InstantiatedModel
    from civic_digital_twins.dt_model.reference_models.molveno.overtourism import (
        M_Base,
    )
    from civic_digital_twins.dt_model.simulation.utils import (
        evaluate_subensemble_worker,
    )

    sub = json.loads(sub_json)
    scen = json.loads(scenario_json)

    # Recreate model inside container
    model = InstantiatedModel(M_Base, values=scen)

    result = evaluate_subensemble_worker(model, scen, sub)

    os.makedirs(os.path.dirname(out_file) or ".", exist_ok=True)
    with open(out_file, "wb") as f:
        pickle.dump(result, f)

    print(f"Saved subensemble result → {out_file}")


# -----------------------------
# SCRIPT: aggregate all results
# -----------------------------
@script(
    image=DEFAULT_IMAGE,
    resources=Resources(memory_request="2Gi"),
)
def aggregate_results(files_json: str, out_file: str):
    """
    files_json: JSON list of file paths to load
    out_file: final aggregated file
    """
    import json, pickle, numpy as np

    paths = json.loads(files_json)

    collected = []
    for p in paths:
        with open(p, "rb") as f:
            collected.append(pickle.load(f))

    # --- perform your aggregation ---
    # Example: sum all fields and confs
    fields = [c["field_batch"] for c in collected]
    field = np.mean(np.stack(fields, axis=0), axis=0)

    # same for confidence if needed
    confs = [c["conf_batch"] for c in collected]
    conf = np.mean(np.stack(confs, axis=0), axis=0)

    out = {
        "field": field,
        "confidence": conf,
        "partials": len(collected),
    }

    with open(out_file, "wb") as f:
        pickle.dump(out, f)

    print(f"FINAL AGGREGATED RESULT → {out_file}")


# -----------------------------
# BUILD WORKFLOW (entrypoint)
# -----------------------------
def build_workflow(
    scenarios_json: str, ensemble_json: str, use_batching: bool, batch_size: int
):

    scenarios = json.loads(scenarios_json)
    ensemble = json.loads(ensemble_json)  # full ensemble list of entries

    with Workflow(
        generate_name="cdt-dag-",
        entrypoint="scenarios",
    ) as w:

        with DAG(name="scenarios"):

            for scen_name, scen_conf in scenarios.items():

                # -----------------------------
                # Split ensemble
                # -----------------------------
                if use_batching:
                    batches = [
                        ensemble[i : i + batch_size]
                        for i in range(0, len(ensemble), batch_size)
                    ]
                else:
                    batches = [[item] for item in ensemble]

                # Convert list of JSONs into parameters for with_param
                batch_params = json.dumps(
                    [
                        {
                            "sub": b,
                            "index": i,
                            "scenario": scen_conf,
                        }
                        for i, b in enumerate(batches)
                    ]
                )

                # -----------------------------
                # Map each batch into a task
                # -----------------------------
                mapped = compute_subensemble.with_param(batch_params)(
                    name=f"{scen_name}-worker",
                    arguments=dict(
                        sub_json="{{item.sub}}",
                        scenario_json="{{item.scenario}}",
                        out_file=f"/tmp/{scen_name}_batch_{{item.index}}.pkl",
                    ),
                )

                # -----------------------------
                # Aggregator (after all workers)
                # -----------------------------
                all_paths = [
                    f"/tmp/{scen_name}_batch_{i}.pkl" for i in range(len(batches))
                ]

                aggregate = aggregate_results(
                    name=f"{scen_name}-aggregate",
                    arguments=dict(
                        files_json=json.dumps(all_paths),
                        out_file=f"/tmp/{scen_name}_final.pkl",
                    ),
                )

                mapped >> aggregate  # dependency

    return w


# --------------------------------
# DigitalHub entrypoint
# --------------------------------
def handler():
    """
    Called by DigitalHub to build the workflow.
    Input parameters are passed via environment variables.
    """
    w = build_workflow(
        scenarios_json=os.environ["SCENARIOS_JSON"],
        ensemble_json=os.environ["ENSEMBLE_JSON"],
        use_batching=os.environ["USE_BATCHING"].lower() == "true",
        batch_size=int(os.environ["BATCH_SIZE"]),
    )

    w.create()
    print("Workflow created successfully.")
