from dataclasses import dataclass
from dataclasses import dataclass
from typing import Any, Dict, Optional

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
    #adaptive tokenization (1 of 2 settings)
    refine_ratio: Optional[float] = None
    imod_bottom: int = 0 #needed only by turbt