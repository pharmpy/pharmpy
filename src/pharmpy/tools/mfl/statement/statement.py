from typing import Union

from .definition import Let
from .feature.feature import ModelFeature

Statement = Union[Let, ModelFeature]
