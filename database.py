from sqlalchemy import create_engine, text
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
import os
from dotenv import load_dotenv

# =====================================================
# 1. Charger les variables d'environnement depuis .env
# =====================================================
load_dotenv()  # <-- IMPORTANT

# =====================================================
# 2. Récupérer les valeurs
# =====================================================
USERNAME = os.getenv("DB_USERNAME")
PASSWORD = os.getenv("DB_PASSWORD")
HOST = os.getenv("DB_HOST")
PORT = os.getenv("DB_PORT")
DB_NAME = os.getenv("DB_NAME")

CA_CERT_PATH = os.path.join(os.path.dirname(__file__), "ca.pem")

# Petit check utile pour le debug local
if not all([USERNAME, PASSWORD, HOST, PORT, DB_NAME]):
    raise RuntimeError("❌ Variables DB_* manquantes. Vérifie ton .env")

# =====================================================
# 3. Créer l'engine temporaire (sans nom de base) pour CREATE DATABASE IF NOT EXISTS
# =====================================================
tmp_engine = create_engine(
    f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/?ssl_ca={CA_CERT_PATH}"
)

with tmp_engine.connect() as conn:
    conn.execute(text(f"CREATE DATABASE IF NOT EXISTS {DB_NAME};"))
    print(f"✅ Base '{DB_NAME}' vérifiée ou créée avec succès.")

# =====================================================
# 4. Engine principal qui pointe sur la base
# =====================================================
SQLALCHEMY_DATABASE_URL = (
    f"mysql+pymysql://{USERNAME}:{PASSWORD}@{HOST}:{PORT}/{DB_NAME}?ssl_ca={CA_CERT_PATH}"
)

engine = create_engine(SQLALCHEMY_DATABASE_URL)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

Base = declarative_base()
