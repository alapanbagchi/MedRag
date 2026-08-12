"""
MedicalConcept
    identify the medical concept
    store its basic name/label
    store identifiers from external sources
    store useful descriptive information
"""
from dataclasses import dataclass, field


@dataclass
class MedicalConcept:
    id: str
    name: str
    description: str
    """
    Holds the identifiers for different biomedical sources. 
    
    Example:
    {
        "mondo": "MONDO:0005252",
        "omim": "209850",
        "orphanet": "106"
    }
    """
    identifiers: dict[str, str] = field(default_factory=dict)
