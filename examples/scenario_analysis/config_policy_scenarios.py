# SPDX-License-Identifier: Apache-2.0
"""Configuration for policy scenarios in the Bologna Mobility simulation."""

# Variants of the model:
BASE_PARAMS = {
    "MODAL_SHIFT_OPTION": "tpm",  ## choice: 'no' or 'tpm'
    "INDUCED_DEMAND_STRATEGY": "none",  ## choice: none, elem_relief
}

# Dict policy init values:
POLICY_PARAMS = {
    "i_p_start_time": "08:00:00",
    "i_p_end_time": "18:00:00",
    "i_p_cost": [5.0 for e in range(7)],
    "i_p_fraction_exempted": 0.0,
    "i_p_pt_frequency_modification": 0.0,
    "i_p_pt_capillarity_modification": 0.0,
    "i_p_pt_cost_modification": 0.0,
    "i_p_pt_time_modification": 0.0,
}

# Dict behaviors init values:
BEHAVIORAL_PARAMS = {
    "i_b_p50_cost": {"loc": 4.00, "scale": 7.00},
    "i_b_p50_anticipating": 1.0,
    "i_b_p50_postponing": 1.0,
    "i_b_p50_anticipation": 1.50,
    "i_b_p50_postponement": 1.50,
    "i_b_pt_capillarity": 4.5,
    "i_b_pt_frequency": -1.45,
    "i_b_pt_cost": -0.30,
    "i_b_pt_time": -0.034,
    "i_b_share_induced_demand": 0.15,
    "i_b_p50_induced_demand": 0.2,
}

# Variation to base params for the scenarios:
scenarios = {
    "A1": {},
    "A2": {
        "policy_params": {
            "i_p_start_time": "07:00:00",
            "i_p_end_time": "19:00:00",
        }
    },
    "A3": {
        "policy_params": {
            "i_p_start_time": "08:00:00",
            "i_p_end_time": "13:00:00",
        }
    },
    "A4": {
        "policy_params": {
            "i_p_start_time": "13:00:00",
            "i_p_end_time": "18:00:00",
        }
    },
    "A5": {
        "policy_params": {
            "i_p_cost": [10.0 for e in range(5)] + [5.0 for e in range(2)],
        }
    },
    "A6": {
        "policy_params": {
            "i_p_fraction_exempted": 0.1,
        }
    },
    "B1": {
        "behavioral_params": {
            "i_b_p50_cost": {"loc": 0.5, "scale": 0.501},
        }
    },
    "B2": {
        "behavioral_params": {
            "i_b_p50_anticipating": 0.1,
            "i_b_p50_postponing": 0.1,
        }
    },
    "B3": {
        "base_params": {
            "MODAL_SHIFT_OPTION": "no",
        }
    },
    "I1": {
        "base_params": {
            "INDUCED_DEMAND_STRATEGY": "elem_relief",
        },
    },
    "I2": {
        "base_params": {
            "INDUCED_DEMAND_STRATEGY": "elem_relief",
        },
        "policy_params": {
            "i_p_start_time": "07:00:00",
            "i_p_end_time": "19:00:00",
        },
    },
    "I3": {
        "base_params": {
            "INDUCED_DEMAND_STRATEGY": "elem_relief",
        },
        "policy_params": {
            "i_p_start_time": "08:00:00",
            "i_p_end_time": "13:00:00",
        },
    },
    "I4": {
        "base_params": {
            "INDUCED_DEMAND_STRATEGY": "elem_relief",
        },
        "policy_params": {
            "i_p_start_time": "13:00:00",
            "i_p_end_time": "18:00:00",
        },
    },
}

group_scenarios = {
    "behavioral_scenarios": ["B1", "B2", "B3", "Base"],
    "policy_duration_scenarios": ["A1", "A3", "A4", "A2", "Base"],
    "policy_fragilities_scenarios": ["A1", "A5", "A6", "Base"],
    "induced_demand_scenarios": ["I1", "I3", "I4", "I2", "Base"],
}
