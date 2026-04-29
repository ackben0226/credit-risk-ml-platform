from pydantic import BaseModel, Field
from typing import Optional, Dict, Any


# =====================================================
# API REQUEST CONTRACT (EXTERNAL - STRICT)
# =====================================================

class PredictionRequest(BaseModel):
    """
    External contract ONLY.
    Must reflect raw incoming data exactly.
    No feature engineering assumptions.
    """

    SK_ID_CURR: int = Field(..., description="Customer ID (tracking only)")

    AMT_INCOME_TOTAL: float
    AMT_CREDIT: float
    AMT_ANNUITY: float

    DAYS_BIRTH: float
    DAYS_EMPLOYED: float

    EXT_SOURCE_1: Optional[float]
    EXT_SOURCE_2: Optional[float]
    EXT_SOURCE_3: Optional[float]


# =====================================================
# RESPONSE CONTRACT (AUDIT-GRADE)
# =====================================================

class PredictionResponse(BaseModel):

    probability: float = Field(..., ge=0.0, le=1.0)
    prediction: int = Field(..., ge=0, le=1)

    model_name: Optional[str]
    model_version: Optional[str]
    request_id: Optional[str]


# =====================================================
# INTERNAL FEATURE CONTRACT (REMOVED FROM Pydantic)
# =====================================================

"""
IMPORTANT DESIGN CHANGE:

❌ DO NOT use Pydantic for model feature schema
✔ use model metadata / registry instead

Reason:
- prevents schema drift
- ensures single source of truth
- avoids duplication risk
"""


# =====================================================
# OBSERVABILITY / AUDIT LOG CONTRACT
# =====================================================

class PredictionLog(BaseModel):

    timestamp: str
    request_id: Optional[str]

    probability: float
    prediction: int

    model_name: Optional[str]
    model_version: Optional[str]

    features: Dict[str, Any]