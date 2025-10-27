from sqlalchemy import Column, Integer, String, Float, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from datetime import datetime
from database import Base

class User(Base):
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)
    username = Column(String(100), nullable=False)
    email = Column(String(255), unique=True, nullable=False)
    password = Column(String(255), nullable=False)
    photo_profil = Column(String(255), nullable=True)

    # ✅ Relation avec les prédictions
    predictions = relationship("Prediction", back_populates="user", cascade="all, delete")

class Prediction(Base):
    __tablename__ = "predictions"

    id = Column(Integer, primary_key=True, index=True)
    id_user = Column(Integer, ForeignKey("users.id"), nullable=False)
    filename = Column(String(255))
    result = Column(String(100))
    confidence = Column(Float)
    prediction_type = Column(String(50))  # 'binary' ou 'multiclass'
    created_at = Column(DateTime, default=datetime.utcnow)

    # ✅ Relation inverse
    user = relationship("User", back_populates="predictions")
