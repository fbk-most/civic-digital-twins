"""Model and index types for digital twin models."""
# SPDX-License-Identifier: Apache-2.0

from ..axes import DOMAIN, ENSEMBLE, PARAMETER, Axis, AxisRole
from .contracts import define, expose, functions, inputs, outputs
from .index import (
    CategoricalIndex,
    ConditionalCategoricalIndex,
    ConditionalDistributionIndex,
    ConstIndex,
    ConstTimeseriesIndex,
    Distribution,
    DistributionIndex,
    DomainValue,
    GenericIndex,
    Index,
    TimeseriesIndex,
)
from .model import (
    AbstractIndexNotInInputsError,
    InputsContractError,
    Model,
    ModelContractError,
    ModelContractViolation,
    ModelContractWarning,
)
from .model_variant import ModelVariant

__all__ = [
    "AbstractIndexNotInInputsError",
    "Axis",
    "define",
    "expose",
    "functions",
    "inputs",
    "outputs",
    "AxisRole",
    "CategoricalIndex",
    "ConditionalCategoricalIndex",
    "ConditionalDistributionIndex",
    "ConstIndex",
    "ConstTimeseriesIndex",
    "Distribution",
    "DistributionIndex",
    "DomainValue",
    "DOMAIN",
    "ENSEMBLE",
    "GenericIndex",
    "Index",
    "InputsContractError",
    "Model",
    "ModelContractError",
    "ModelContractViolation",
    "ModelContractWarning",
    "ModelVariant",
    "PARAMETER",
    "TimeseriesIndex",
]
