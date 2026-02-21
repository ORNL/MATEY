# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: 2026 UT-Battelle, LLC
# This file is part of the MATEY Project.

from dataclasses import dataclass
from typing import Any, Dict, Optional
from torch import Tensor

@dataclass
class TrainOptionsBase:
    returnbase4train: bool = False
    
@dataclass
class ForwardOptionsBase:
    #always passed
    imod: int = 0
    tkhead_name: Optional[str] = None
    leadtime: Optional[int]|None = None  
    #optional
    sequence_parallel_group: Any|None = None  
    blockdict: Optional[Dict[str, Any]] = None
    cond_dict: Optional[Dict[str, Any]] = None
    cond_input: Optional[Tensor] = None
    tkhead_type: Optional[str] = 'default'     # 'default', 'graph'
    field_labels_out: Optional[Tensor] = None
    #adaptive tokenization (1 of 2 settings)
    refine_ratio: Optional[float] = None
    imod_bottom: int = 0 #needed only by turbt
