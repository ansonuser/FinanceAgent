from dataclasses import dataclass
from typing import Dict

@dataclass
class RevenueData:
    period: str
    currency: str
    unit: str
    product_segments: Dict[str, int]
    total_revenue: int
    reasoning: str

