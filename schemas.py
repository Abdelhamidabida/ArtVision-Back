from pydantic import BaseModel, EmailStr, Field
from typing import Optional, List
from datetime import datetime

# ========================
# 🔹 Schémas Utilisateur
# ========================

class UserBase(BaseModel):
    username: str
    email: EmailStr
    photo_profil: Optional[str] = None

class UserCreate(UserBase):
    password: str = Field(..., min_length=6)

class UserUpdate(BaseModel):
    username: Optional[str] = None
    email: Optional[EmailStr] = None
    password: Optional[str] = None
    photo_profil: Optional[str] = None

# ========================
# 🔹 Schémas Prediction
# ========================

class PredictionBase(BaseModel):
    filename: str
    result: str
    confidence: float
    prediction_type: str

class PredictionCreate(PredictionBase):
    id_user: int

class PredictionResponse(PredictionBase):
    id: int
    created_at: datetime

    class Config:
        orm_mode = True

# ========================
# 🔹 Schéma User (avec historique de prédictions)
# ========================

class UserResponse(UserBase):
    id: int
    predictions: List[PredictionResponse] = []  # historique des prédictions

    class Config:
        orm_mode = True
