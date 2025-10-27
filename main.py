from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Optional
import numpy as np
import io
import os
import tempfile
import torch
from PIL import Image, UnidentifiedImageError
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
from transformers import BlipProcessor, BlipForConditionalGeneration
import cloudinary.uploader

from database import SessionLocal, engine, Base
from models import User, Prediction
from schemas import UserResponse, UserUpdate, PredictionResponse
from auth import create_access_token, verify_password, hash_password
from config_cloudinary import cloudinary

# =====================================================
# 🚀 INITIALISATION DE L’API
# =====================================================
app = FastAPI(title="ArtVision API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚠ Tu peux restreindre à ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Création des tables si elles n'existent pas
Base.metadata.create_all(bind=engine)

# =====================================================
# 🔌 DB Session
# =====================================================
def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =====================================================
# 🧠 CONFIG CHEMINS MODÈLES
# =====================================================
MODEL_BINARY_PATH = "models/binary_best_df.h5"
MODEL_MULTI_PATH = "models/multi_balanced_best.h5"


CATEGORIES = ["Painting", "Photo", "Schematic", "Sketch", "Text"]



# =====================================================
# 🌟 CHARGEMENT DES MODÈLES AU DÉMARRAGE
# =====================================================
@app.on_event("startup")
def load_all_models():
    """
    ⚠️ IMPORTANT POUR RENDER :
    On charge les modèles ici (après que le serveur démarre),
    pas au niveau global, pour que Render voie le port ouvert.
    """
    print("🚀 [STARTUP] Chargement des modèles IA...")

    # Models Classification (TensorFlow)
    app.state.binary_model = load_model(MODEL_BINARY_PATH)
    app.state.multi_model = load_model(MODEL_MULTI_PATH)

    
    # BLIP Captioning (Hugging Face)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    app.state.device = device
    app.state.blip_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    app.state.blip_model = BlipForConditionalGeneration.from_pretrained(
        "Salesforce/blip-image-captioning-base"
    ).to(device)

    print(f"✅ [STARTUP] Modèles chargés. BLIP sur {device.upper()}.")

# =====================================================
# 🔹 ROUTES UTILISATEURS
# =====================================================

@app.post("/users/", response_model=UserResponse)
async def create_user(
    username: str = Form(...),
    email: str = Form(...),
    password: str = Form(...),
    photo_profil: Optional[UploadFile] = File(None),
    db: Session = Depends(get_db)
):
    # Vérifier doublon email
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email déjà utilisé")

    hashed_pw = hash_password(password)

    # Upload Cloudinary (photo profil)
    photo_url = None
    if photo_profil:
        upload_result = cloudinary.uploader.upload(
            photo_profil.file,
            folder="artvision/profile_photos"
        )
        photo_url = upload_result.get("secure_url")

    # Création user
    new_user = User(username=username, email=email, password=hashed_pw, photo_profil=photo_url)
    db.add(new_user)
    db.commit()
    db.refresh(new_user)
    return new_user

@app.get("/users/", response_model=list[UserResponse])
def get_users(db: Session = Depends(get_db)):
    return db.query(User).all()

@app.get("/users/{user_id}", response_model=UserResponse)
def get_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    return user

@app.put("/users/{user_id}", response_model=UserResponse)
def update_user(user_id: int, user_update: UserUpdate, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")

    for key, value in user_update.dict(exclude_unset=True).items():
        if key == "password":
            # hash du nouveau mot de passe
            import bcrypt
            value = bcrypt.hashpw(value.encode('utf-8'), bcrypt.gensalt()).decode('utf-8')
        setattr(user, key, value)

    db.commit()
    db.refresh(user)
    return user

@app.delete("/users/{user_id}")
def delete_user(user_id: int, db: Session = Depends(get_db)):
    user = db.query(User).filter(User.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")
    db.delete(user)
    db.commit()
    return {"message": "Utilisateur supprimé"}

# =====================================================
# 🔐 LOGIN UTILISATEUR
# =====================================================

@app.post("/login")
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.email == form_data.username).first()
    if not user:
        raise HTTPException(status_code=400, detail="Utilisateur introuvable")

    if not verify_password(form_data.password, user.password):
        raise HTTPException(status_code=400, detail="Mot de passe incorrect")

    access_token = create_access_token(data={"sub": user.email})

    user_predictions = db.query(Prediction).filter(Prediction.id_user == user.id).all()

    return {
        "access_token": access_token,
        "token_type": "bearer",
        "user": {
            "id": user.id,
            "username": user.username,
            "email": user.email,
            "photo_profil": user.photo_profil,
            "predictions": [
                {
                    "id": p.id,
                    "filename": p.filename,
                    "result": p.result,
                    "confidence": p.confidence,
                    "prediction_type": p.prediction_type,
                    "created_at": p.created_at
                } for p in user_predictions
            ]
        }
    }

# =====================================================
# 🔮 ROUTE PRÉDICTION (binaire / multiclass)
# =====================================================

@app.post("/predictions/", response_model=PredictionResponse)
async def create_prediction(
    request: Request,
    id_user: int = Form(...),
    prediction_type: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    # Vérif utilisateur
    user = db.query(User).filter(User.id == id_user).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")

    # Lire l'image
    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predicted_class = None
    confidence = None

    # Récupération des modèles chargés en startup
    binary_model = request.app.state.binary_model
    multi_model = request.app.state.multi_model

    if prediction_type.lower() == "binaire":
        pred = binary_model.predict(img_array, verbose=0)[0][0]
        predicted_class = "Photo" if pred > 0.5 else "Non-photo"
        confidence = float(pred if pred > 0.5 else 1 - pred)

    elif prediction_type.lower() == "multiclass":
        pred = multi_model.predict(img_array, verbose=0)[0]
        predicted_class = CATEGORIES[int(np.argmax(pred))]
        confidence = float(np.max(pred))

    else:
        raise HTTPException(
            status_code=400,
            detail="prediction_type doit être 'binaire' ou 'multiclass'"
        )

    # Upload image entrante
    file.file.seek(0)
    upload_result = cloudinary.uploader.upload(file.file, folder="artvision/predictions")
    image_url = upload_result["secure_url"]

    # Sauvegarde en base
    new_pred = Prediction(
        id_user=id_user,
        filename=image_url,
        result=predicted_class,
        confidence=confidence,
        prediction_type=prediction_type,
        created_at=datetime.utcnow()
    )
    db.add(new_pred)
    db.commit()
    db.refresh(new_pred)

    return new_pred

@app.get("/predictions/user/{id_user}", response_model=list[PredictionResponse])
def get_predictions_for_user(id_user: int, db: Session = Depends(get_db)):
    return db.query(Prediction).filter(Prediction.id_user == id_user).all()



# =====================================================
# 📝 CAPTIONING (BLIP)
# =====================================================

@app.post("/caption", response_model=PredictionResponse)
async def caption_image(
    request: Request,
    id_user: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Génère une légende textuelle à partir d'une image via BLIP.
    Sauvegarde aussi la prédiction en base.
    """
    try:
        # Vérifier utilisateur
        user = db.query(User).filter(User.id == id_user).first()
        if not user:
            raise HTTPException(status_code=404, detail="Utilisateur non trouvé")

        # Lire l'image
        contents = await file.read()
        pil_img = Image.open(io.BytesIO(contents)).convert("RGB")

        # Récupérer le modèle BLIP + processor
        processor = request.app.state.blip_processor
        blip_model = request.app.state.blip_model
        device = request.app.state.device

        # Générer caption
        inputs = processor(images=pil_img, return_tensors="pt").to(device)
        output = blip_model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(output[0], skip_special_tokens=True)

        # Upload image dans Cloudinary
        file.file.seek(0)
        upload_result = cloudinary.uploader.upload(file.file, folder="artvision/captions")
        image_url = upload_result["secure_url"]

        # Enregistrer la prédiction dans la base
        new_pred = Prediction(
            id_user=id_user,
            filename=image_url,
            result=caption,
            confidence=1.0,
            prediction_type="captioning",
            created_at=datetime.utcnow()
        )

        db.add(new_pred)
        db.commit()
        db.refresh(new_pred)

        return new_pred

    except Exception as e:
        print("❌ Erreur dans /caption :", e)
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# 🧪 HEALTHCHECK
# =====================================================

@app.get("/test")
def test():
    return {"message": "✅ API opérationnelle avec classification + débruitage + captioning"}
