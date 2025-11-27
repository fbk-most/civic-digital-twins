"""
Workflow for parallel ensemble evaluation.

Runtime expected run parameters (passed when starting a pipeline run):
 - SCENARIO_NAME: str
 - SCENARIO_JSON: str (JSON-serialized scenario config)
 - PARAMS_JSON: str (JSON-serialized params dict)

Notes:
 - The workflow uses civic_digital_twins package. Ensure requirements include it.
 - The builder will create a fixed number of worker tasks (ensemble_size),
   each checks whether its batch index exists in the manifest and processes only if present.
"""

import os

from hera.workflows import Workflow, DAG, script, Resources

DEFAULT_IMAGE = os.environ.get("DH_WORKFLOW_IMAGE", "python:3.12-slim")
ENSEMBLE_LIMIT = 5

# -------------------------
# Task: split (create batches manifest)
# -------------------------
@script(image=DEFAULT_IMAGE, resources=Resources(memory_request="1Gi"))
def split_task(scenario_json: str, params_json: str, out_dir: str):
    """
    Creates manifest.json with:
      - scenario_hash
      - batches: list of batches, each containing items of the ensemble
    Batching rule:
      - If ensemble_size <= ENSEMBLE_LIMIT  -> one element per batch
      - If ensemble_size > ENSEMBLE_LIMIT   -> batches of size ENSEMBLE_LIMIT
    """
    import os, json
    from civic_digital_twins.dt_model.engine import InstantiatedModel
    from civic_digital_twins.dt_model.reference_models.molveno.overtourism import M_Base
    from civic_digital_twins.dt_model.model.ensemble import Ensemble

    scen = json.loads(scenario_json)
    params = json.loads(params_json)

    # Build the model
    inst = InstantiatedModel(M_Base, values=scen)

    # Build full ensemble (list of (prob, cv_values))
    ensemble_obj = Ensemble(inst, scen, cv_ensemble_size=params.get("ensemble_size", 20))
    ensemble_list = list(ensemble_obj)

    ensemble_size = len(ensemble_list)

    # --- NEW BATCHING RULE ---
    if ensemble_size <= ENSEMBLE_LIMIT:
        batch_size = 1
    else:
        batch_size = ENSEMBLE_LIMIT
    # --------------------------

    # Generate batches
    batches = [
        ensemble_list[i : i + batch_size]
        for i in range(0, ensemble_size, batch_size)
    ]

    # Compute scenario hash (as before)
    scenario_hash = ensemble_obj.compute_hash(
        [
            params.get("t_max"),
            params.get("e_max"),
            params.get("t_sample"),
            params.get("e_sample"),
            params.get("target_presence_samples"),
        ]
    )

    # Write manifest
    os.makedirs(out_dir, exist_ok=True)
    manifest = {
        "scenario_hash": scenario_hash,
        "ensemble_size": ensemble_size,
        "batch_size": batch_size,
        "batches_count": len(batches),
        "batches": batches,
    }

    with open(os.path.join(out_dir, "manifest.json"), "w") as f:
        json.dump(manifest, f)

    print(
        f"[split_task] ensemble_size={ensemble_size}, "
        f"batch_size={batch_size}, batches={len(batches)}"
    )


# -------------------------
# Task: worker (compute a batch index if present)
# -------------------------
@script(image=DEFAULT_IMAGE, resources=Resources(memory_request="2Gi"))
def worker_task(batch_index: int, out_dir: str, scenario_json: str, params_json: str):
    """
    Each worker checks out_dir/manifest.json for its batch index.
    If present: computes field_batch and conf_batch and saves to out_dir/batch_{idx}.pkl
    If not present: exits (no-op).
    """
    import os, json, pickle
    import numpy as np

    from civic_digital_twins.dt_model.engine import InstantiatedModel
    from civic_digital_twins.dt_model.reference_models.molveno.overtourism import M_Base
    from civic_digital_twins.dt_model.simulation.utils import evaluate_subensemble_worker

    manifest_path = os.path.join(out_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        print(f"[worker:{batch_index}] manifest not found -> exiting")
        return

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    batches = manifest.get("batches", [])
    if batch_index >= len(batches):
        print(f"[worker:{batch_index}] no batch for this index -> exiting")
        return

    # Extract the sub-ensemble
    sub = batches[batch_index]
    scen = json.loads(scenario_json)
    params = json.loads(params_json)

    # Recreate model and evaluate sub-ensemble
    model = InstantiatedModel(M_Base, values=scen)

    # We expect evaluate_subensemble_worker to return a dict with 'field_batch' and 'conf_batch'
    batch_result = evaluate_subensemble_worker(model, scen, sub, params=params)

    out_path = os.path.join(out_dir, f"batch_{batch_index}.pkl")
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        pickle.dump(batch_result, f)

    print(f"[worker:{batch_index}] wrote {out_path}")


# -------------------------
# Task: aggregator (load existing batch pickles and aggregate)
# -------------------------
@script(image=DEFAULT_IMAGE, resources=Resources(memory_request="2Gi"))
def aggregator_task(out_dir: str, scenario_name: str):
    """
    Read manifest.json, gather available batch_{i}.pkl files, aggregate them,
    write final {scenario_name}_{scenario_hash}.pkl and log artifact using DigitalHub SDK.
    """
    import os, json, pickle
    import numpy as np
    import digitalhub as dh

    manifest_path = os.path.join(out_dir, "manifest.json")
    if not os.path.exists(manifest_path):
        raise RuntimeError("Manifest missing in aggregator")

    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    scenario_hash = manifest.get("scenario_hash", "nohash")
    batches_count = manifest.get("batches_count", 0)

    # Collect existing batch pickles
    collected = []
    for i in range(batches_count):
        p = os.path.join(out_dir, f"batch_{i}.pkl")
        if not os.path.exists(p):
            print(f"[aggregator] missing batch file {p} (skipping)")
            continue
        with open(p, "rb") as f:
            collected.append(pickle.load(f))

    if len(collected) == 0:
        final = {"field": None, "confidence": None, "partials": 0, "params": None}
    else:
        fields = [c["field_batch"] for c in collected]
        confs = [c["conf_batch"] for c in collected]

        field = np.mean(np.stack(fields, axis=0), axis=0)
        conf = np.mean(np.stack(confs, axis=0), axis=0)

        final = {
            "field": field,
            "confidence": conf,
            "partials": len(collected),
            "params": manifest.get("params", None),
        }

    # Save final pkl using {scenario_name}_{scenario_hash}.pkl
    filename = f"{scenario_name}_{scenario_hash}.pkl"
    out_path = os.path.join(out_dir, filename)

    with open(out_path, "wb") as f:
        pickle.dump(final, f)

    print(f"[aggregator] final saved to {out_path}")

    # Log artifact to DigitalHub
    try:
        project_name = os.environ.get("DH_PROJECT", "cdt-parallelization")
        project = dh.get_project(project_name)
        project.log_artifact(kind="artifact", name=filename, source=out_path)
        print(f"[aggregator] logged artifact {filename} to project {project_name}")
    except Exception as e:
        print(f"[aggregator] WARNING: failed to log artifact: {e}")


# -------------------------
# Build the workflow
# -------------------------
def build_workflow():
    """
    Build workflow once (registration time). We will create a fixed number of worker tasks
    equal to ensemble_size (safe upper bound). The split task will create the manifest at run-time.
    """
    # Read package-default ensemble_size to decide how many worker tasks to create
    try:
        from civic_digital_twins.dt_model.model.setup import params as default_params
        ensemble_size_default = default_params.get("ensemble_size", 20)
    except Exception:
        ensemble_size_default = 20

    max_workers = int(ensemble_size_default)  # safe upper bound

    with Workflow(generate_name="cdt-parallel-", entrypoint="pipeline") as w:
        with DAG(name="pipeline"):
            # per-run unique folder (workflow.uid is available at run-time inside pods)
            out_dir = "/tmp/{{workflow.uid}}"

            # split task: will materialize ensemble at run time inside the pod
            split = split_task(
                name="split",
                arguments=dict(
                    scenario_json="{{workflow.parameters.SCENARIO_JSON}}",
                    params_json="{{workflow.parameters.PARAMS_JSON}}",
                    out_dir=out_dir,
                ),
            )

            # Create max_workers worker tasks (they will read manifest and decide whether to work)
            workers = []
            for idx in range(max_workers):
                wtask = worker_task(
                    name=f"worker-{idx}",
                    arguments=dict(
                        batch_index=idx,
                        out_dir=out_dir,
                        scenario_json="{{workflow.parameters.SCENARIO_JSON}}",
                        params_json="{{workflow.parameters.PARAMS_JSON}}",
                    ),
                )
                # ensure split finishes before any worker
                split >> wtask
                workers.append(wtask)

            # aggregator depends on all workers
            agg = aggregator_task(
                name="aggregator",
                arguments=dict(
                    out_dir=out_dir,
                    scenario_name="{{workflow.parameters.SCENARIO_NAME}}",
                ),
            )

            for wtask in workers:
                wtask >> agg

    return w


# Hera/DigitalHub entrypoint used at registration time
def handler():
    w = build_workflow()
    w.create()
