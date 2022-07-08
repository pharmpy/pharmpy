from typing import Union

from .definition import Definition
from .feature.feature import ModelFeature

Statement = Union[Definition, ModelFeature]
