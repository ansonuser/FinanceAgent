from dataclasses import dataclass
from typing import Dict

@dataclass
class RevenueData:
    period: str
    currency: str
    scale: str
    product_segments: Dict[str, float]
    total_revenue: int
    reasoning: str

@dataclass
class CompanyIndexEntry:
    cik: str
    ticker: str
    name: str