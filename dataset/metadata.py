import json
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional

@dataclass
class CaseMetadata:
    case_id: str
    architecture: str
    difficulty: str
    seed: int
    search_width: int
    search_height: int
    reference_width: int
    reference_height: int
    nominal_scale: float
    augmentation_scale: float
    effective_scale: float
    rotation_degrees: float
    gt_x: float
    gt_y: float
    gt_bbox: Dict[str, float]
    reference_noise_level: float
    search_noise_level: float
    blur_kernel: int
    edge_brightening_strength: float
    pitch_x: int
    pitch_y: int
    feature_width: int
    feature_height: int
    
    # V2 Metadata Fields
    reference_context_type: Optional[str] = None
    macro_boundary_present: Optional[bool] = None
    periodicity_score: Optional[float] = None
    edge_density: Optional[float] = None
    gradient_energy: Optional[float] = None
    local_variance: Optional[float] = None
    reference_sampling_rule: Optional[str] = None
    reference_origin_x: Optional[int] = None
    reference_origin_y: Optional[int] = None
    
    # V3 Metadata Fields
    spatial_region: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)

    def save_json(self, path: str):
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(self.to_dict(), f, indent=4)
