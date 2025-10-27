from fastapi import FastAPI, HTTPException, UploadFile, File, Form, Depends, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from tensorflow.keras.models import load_model
from tensorflow.keras import layers, models
import numpy as np
import io
import tensorflow as tf
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
from auth import create_access_token, verify_password, hash_password
from io import BytesIO
from PIL import Image, UnidentifiedImageError
from database import SessionLocal, engine, Base
from schemas import UserResponse, UserUpdate, PredictionResponse
from sqlalchemy.orm import Session
from models import User, Prediction
import bcrypt
import os
from datetime import datetime
from config_cloudinary import cloudinary
import cloudinary.uploader
from fastapi.security import OAuth2PasswordRequestForm
from typing import Optional
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json
import json
# =====================================================
# 🚀 INITIALISATION DE L’API
# =====================================================
app = FastAPI(title="ArtVision API", version="2.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # ⚙️ tu peux restreindre à ["http://localhost:5173"]
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

Base.metadata.create_all(bind=engine)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# =====================================================
# 🧠 CHARGEMENT DES MODÈLES
# =====================================================
MODEL_BINARY_PATH = "models/binary_best_df.h5"
MODEL_MULTI_PATH = "models/multi_balanced_best.h5"


print("📦 Chargement des modèles...")
binary_model = load_model(MODEL_BINARY_PATH)
multi_model = load_model(MODEL_MULTI_PATH)
print("✅ Modèles de classification chargés avec succès !")





# =====================================================
# 🏷️ CLASSES MULTI-CLASSES
# =====================================================
CATEGORIES = ["Painting", "Photo", "Schematic", "Sketch", "Text"]

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
    if db.query(User).filter(User.email == email).first():
        raise HTTPException(status_code=400, detail="Email déjà utilisé")

    hashed_pw = hash_password(password)

    photo_url = None
    if photo_profil:
        upload_result = cloudinary.uploader.upload(
            photo_profil.file,
            folder="artvision/profile_photos"
        )
        photo_url = upload_result.get("secure_url")

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
# 🔹 LOGIN UTILISATEUR
# =====================================================
@app.post("/login")
def login(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
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
# 🔹 ROUTES PRÉDICTIONS (BINAIRE / MULTI / DENOISING)
# =====================================================
@app.post("/predictions/", response_model=PredictionResponse)
async def create_prediction(
    id_user: int = Form(...),
    prediction_type: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    user = db.query(User).filter(User.id == id_user).first()
    if not user:
        raise HTTPException(status_code=404, detail="Utilisateur non trouvé")

    contents = await file.read()
    image = Image.open(io.BytesIO(contents)).convert("RGB")
    image = image.resize((224, 224))
    img_array = np.array(image) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    predicted_class, confidence = None, None

    if prediction_type.lower() == "binaire":
        pred = binary_model.predict(img_array, verbose=0)[0][0]
        predicted_class = "Photo" if pred > 0.5 else "Non-photo"
        confidence = float(pred if pred > 0.5 else 1 - pred)

    elif prediction_type.lower() == "multiclass":
        pred = multi_model.predict(img_array, verbose=0)[0]
        predicted_class = CATEGORIES[int(np.argmax(pred))]
        confidence = float(np.max(pred))

   
    else:
        raise HTTPException(status_code=400, detail="prediction_type doit être 'binaire', 'multiclass' ou 'denoising'")

    # Upload l'image originale si pas débruitage
    if prediction_type.lower() != "denoising":
        file.file.seek(0)
        upload_result = cloudinary.uploader.upload(file.file, folder="artvision/predictions")
        image_url = upload_result["secure_url"]

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
# 🔹 ROUTE DÉBRUITAGE (ENREGISTRE EN BASE)
# =====================================================
@app.post("/predictDenoising", response_model=PredictionResponse)
async def predict_denoising(
    id_user: int = Form(...),
    prediction_type: str = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    🔹 Route de débruitage d'image :
    - Prétraitement automatique
    - Prédiction du modèle
    - Correction de normalisation
    - Sauvegarde Cloudinary + base
    """
    try:
        # ============================
        # Vérification utilisateur
        # ============================
        user = db.query(User).filter(User.id == id_user).first()
        if not user:
            raise HTTPException(status_code=404, detail="Utilisateur non trouvé")

        # ============================
        # Lecture de l'image
        # ============================
        contents = await file.read()
        img = Image.open(BytesIO(contents)).convert("RGB").resize((224, 224))
        img_array = np.array(img, dtype=np.float32)

        # ============================
        # Normalisation (entrée)
        # ============================
        # ⚠️ Test automatique : on essaie les deux méthodes
        img_array_norm = img_array / 255.0
        img_array_norm = np.expand_dims(img_array_norm, axis=0)

        # ============================
        # Prédiction
        # ============================
        print("🧠 [DEBUG] Prédiction en cours...")
        denoised = denoising_model.predict(img_array_norm, verbose=0)[0]

        print(f"📊 Sortie modèle → min={denoised.min():.4f}, max={denoised.max():.4f}, mean={denoised.mean():.4f}")

        # ============================
        # Post-traitement
        # ============================
        # Si le modèle sort dans [-1,1], on le remet dans [0,1]
        if denoised.min() < 0:
            print("🔄 Sortie détectée dans [-1,1] → Conversion vers [0,1]")
            denoised = (denoised + 1.0) / 2.0

        # Clamp des valeurs
        denoised = np.clip(denoised, 0, 1)

        # Conversion en uint8
        denoised_uint8 = (denoised * 255).astype(np.uint8)

        print(f"✅ Après normalisation → min={denoised_uint8.min()}, max={denoised_uint8.max()}, mean={denoised_uint8.mean():.2f}")

        # ============================
        # Test visuel local (DEBUG)
        # ============================
        local_debug_path = "debug_denoised_local.png"
        Image.fromarray(denoised_uint8).save(local_debug_path)
        print(f"🖼️ Image locale sauvegardée → {local_debug_path}")

        # ============================
        # Sauvegarde temporaire + Cloudinary
        # ============================
        import tempfile, os
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            temp_path = tmp.name
            Image.fromarray(denoised_uint8).save(temp_path)

        # 📤 Upload sur Cloudinary
        upload_result = cloudinary.uploader.upload(
            temp_path,
            folder="artvision/denoised",
            resource_type="image"
        )
        image_url = upload_result["secure_url"]

        try:
            os.remove(temp_path)
        except Exception as err:
            print(f"⚠️ Impossible de supprimer {temp_path}: {err}")

        # ============================
        # Enregistrement en base
        # ============================
        new_pred = Prediction(
            id_user=id_user,
            filename=image_url,
            result="Image débruitée avec succès",
            confidence=1.0,
            prediction_type=prediction_type,
            created_at=datetime.utcnow()
        )

        db.add(new_pred)
        db.commit()
        db.refresh(new_pred)

        print("✅ Débruitage terminé avec succès !")
        return new_pred

    except UnidentifiedImageError:
        raise HTTPException(status_code=400, detail="Le fichier fourni n'est pas une image valide.")
    except Exception as e:
        print("❌ Erreur dans /predictDenoising :", e)
        raise HTTPException(status_code=500, detail=str(e))


# =====================================================
# 🔹 TEST
# =====================================================
@app.get("/test")
def test():
    return {"message": "✅ API opérationnelle avec classification + débruitage"}

# =====================================================
# 🔹 Captioning avec Salesforce BLIP (modèle pré-entraîné Hugging Face)
# =====================================================

from transformers import BlipProcessor, BlipForConditionalGeneration
from PIL import Image
import torch
import io
from datetime import datetime
from fastapi import Form, File, UploadFile, HTTPException, Depends

# =====================================================
# 🧠 Chargement du modèle BLIP
# =====================================================
print("📦 Chargement du modèle 'Salesforce/blip-image-captioning-base'...")
device = "cuda" if torch.cuda.is_available() else "cpu"

processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base").to(device)

print(f"✅ Modèle BLIP chargé sur {device.upper()}")

# =====================================================
# 🧠 Fonction de génération de légende avec BLIP
# =====================================================
def generate_caption_blip(image: Image.Image) -> str:
    try:
        inputs = processor(images=image, return_tensors="pt").to(device)
        output = model.generate(**inputs, max_new_tokens=30)
        caption = processor.decode(output[0], skip_special_tokens=True)
        return caption
    except Exception as e:
        print(f"❌ Erreur lors de la génération du caption BLIP: {e}")
        raise HTTPException(status_code=500, detail=str(e))

# =====================================================
# 🔹 Endpoint FastAPI : /caption
# =====================================================
@app.post("/caption")
async def caption_image(
    id_user: int = Form(...),
    file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    """
    Génère une légende (caption) à partir d'une image avec BLIP.
    """
    try:
        # Vérifier user
        user = db.query(User).filter(User.id == id_user).first()
        if not user:
            raise HTTPException(status_code=404, detail="Utilisateur non trouvé")

        # Lire l'image
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")

        # Générer le caption via BLIP
        caption = generate_caption_blip(image)
        print(f"🧠 Caption généré : {caption}")

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


