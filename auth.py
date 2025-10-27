from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from sqlalchemy.orm import Session
from database import SessionLocal
from models import User
from dotenv import load_dotenv
import os

# =====================================================
# 🔐 CONFIGURATION JWT
# =====================================================
load_dotenv()  # charge les variables d’environnement depuis .env

SECRET_KEY = os.getenv("SECRET_KEY", "changeme_secret_key")
ALGORITHM = os.getenv("ALGORITHM", "HS256")
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", 60))

# Gestion du hachage des mots de passe
# ✅ Remplace bcrypt par pbkdf2_sha256 (aucune limite de 72 caractères, stable sous Windows)
pwd_context = CryptContext(schemes=["pbkdf2_sha256"], deprecated="auto")

# Configuration OAuth2 (token transmis via le header Authorization)
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/login")

# =====================================================
# 🔹 FONCTIONS UTILITAIRES (hash, vérification, token)
# =====================================================
def hash_password(password: str) -> str:
    return pwd_context.hash(password)



def verify_password(plain_password: str, hashed_password: str) -> bool:
    """
    Vérifie qu’un mot de passe en clair correspond à son hash.
    """
    return pwd_context.verify(plain_password, hashed_password)


def create_access_token(data: dict, expires_delta: timedelta | None = None) -> str:
    """
    Crée un JWT signé avec une date d’expiration.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

# =====================================================
# 🔹 GESTION DE LA BASE DE DONNÉES
# =====================================================
def get_db():
    """
    Dépendance pour obtenir une session SQLAlchemy.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# =====================================================
# 🔹 RÉCUPÉRER L’UTILISATEUR CONNECTÉ
# =====================================================
def get_current_user(token: str = Depends(oauth2_scheme), db: Session = Depends(get_db)) -> User:
    """
    Décode le token JWT et renvoie l’utilisateur correspondant.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Token invalide ou expiré",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Décoder le token JWT
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        email: str = payload.get("sub")
        if email is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception

    # Vérifie si l'utilisateur existe
    user = db.query(User).filter(User.email == email).first()
    if user is None:
        raise credentials_exception

    return user
