# SPDX-License-Identifier: Apache-2.0
"""Pytest session configuration."""

import warnings

# Suppress the DeprecationWarning emitted by Model.__init_subclass__ for
# classes that define __init__ directly.  Test models routinely use the old
# pattern; tagging each one legacy=True would be mechanical noise.  The filter
# is scoped to the exact message so that other deprecation warnings (e.g. the
# @dataclass-for-inputs warning asserted in test_io_contracts.py) are unaffected.
warnings.filterwarnings(
    "ignore",
    message=r".*Use @define with compute\(\).*",
    category=DeprecationWarning,
)
