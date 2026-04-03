# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

from .data import *
from .extractors import *
from .model import *


def __getattr__(name):
    if name == "ClassificationExperiment":
        from .experiment import ClassificationExperiment

        return ClassificationExperiment
    if name == "ClassificationModule":
        from .pl_module import ClassificationModule

        return ClassificationModule
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
