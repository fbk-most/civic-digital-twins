<!-- SPDX-License-Identifier: Apache-2.0 -->

# Bologna Mobility Scenario Analysis

Policy scenario analysis for the Bologna Mobility cordon-charging model.

## Description

This example runs multiple policy scenarios (extended hours, higher fees, exemptions,
modal-shift, induced demand) through the Bologna Mobility simulation and reports
global KPIs (inflow, traffic, emissions, fees, time/mode shifts).

## Files

| File | Purpose |
|------|---------|
| `bologna_mobility_simulation.py` | Core simulation model |
| `bologna_mobility_data.py` | Embedded time-series and euro-class data |
| `config_policy_scenarios.py` | Scenario definitions and behavioural parameters |
| `test_policy_scenarios.py` | Entry point that runs all scenarios and saves KPIs |
| `expected_global_kpis_scenarios.csv` | Reference KPIs for consistency checks |

## Usage

From the repository root:

```bash
# Run all scenarios, save KPIs to output/kpi/
uv run python examples/scenario_analysis/test_policy_scenarios.py

# Also generate time-series plots to output/img/
uv run python examples/scenario_analysis/test_policy_scenarios.py --plot
```

## Output

```
examples/scenario_analysis/output/
├── kpi/
│   └── global_kpis_scenarios.csv   # one column per scenario, one row per KPI
└── img/                            # only when --plot is passed
    ├── plot_<group>_<quantity>.png
    └── plot_u_<group>_<name>_<quantity>.png
```

## Consistency check

Compare emitted KPIs against the expected reference:

```bash
uv run python -c "
import pandas as pd
got = pd.read_csv('examples/scenario_analysis/output/kpi/global_kpis_scenarios.csv', index_col=0)
exp = pd.read_csv('examples/scenario_analysis/expected_global_kpis_scenarios.csv', index_col=0)
pd.testing.assert_frame_equal(got, exp, check_exact=False, atol=0.6)
print('OK – KPIs match within tolerance')
"
```

A tolerance of `atol=0.6` accounts for minor floating-point differences across
platforms. Increase it if needed.
