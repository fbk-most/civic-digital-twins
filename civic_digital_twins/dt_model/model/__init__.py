"""Model and index types for digital twin models."""
# SPDX-License-Identifier: Apache-2.0

from ..axes import DOMAIN, ENSEMBLE, PARAMETER, Axis, AxisRole
from .contracts import define, expose, functions, inputs, outputs
from .index import (
    CategoricalIndex,
    ConditionalCategoricalIndex,
    ConditionalDistributionIndex,
    ConstDomainIndex,
    ConstIndex,
    ConstTimeseriesIndex,
    Distribution,
    DistributionIndex,
    DomainIndex,
    DomainValue,
    GenericIndex,
    Index,
    TimeseriesIndex,
)
from .model import (
    AbstractIndexNotInInputsError,
    FunctionsTypeMismatchError,
    InputsContractError,
    InputsTypeMismatchError,
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
    "ConstDomainIndex",
    "ConstIndex",
    "ConstTimeseriesIndex",
    "Distribution",
    "DistributionIndex",
    "DomainIndex",
    "DomainValue",
    "DOMAIN",
    "ENSEMBLE",
    "FunctionsTypeMismatchError",
    "GenericIndex",
    "Index",
    "InputsContractError",
    "InputsTypeMismatchError",
    "Model",
    "ModelContractError",
    "ModelContractViolation",
    "ModelContractWarning",
    "ModelVariant",
    "PARAMETER",
    "TimeseriesIndex",
]
