import math
from dataclasses import dataclass, field
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .base import BaseModelArgs, BaseModelOutput, normalize_embeddings
from .pooling import pool_by_config


@dataclass
class ModelArgs(BaseModelArgs):
    model_type: str
    hidden_size: int
    num_hidden_layers: int