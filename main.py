# app/main.py
import os, json
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List

from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm, OAuth2PasswordBearer
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from sqlalchemy import create_engine, Column, Integer, String, Boolean, DateTime, JSON, ForeignKey
from sqlalchemy.orm import sessionmaker, declarative_base, relationship
from passlib.context import CryptContext
from jose import jwt, JWTError
import joblib

from dotenv import load_dotenv
load_dotenv()

# --- Config ---
SECRET_KEY = os.getenv("SECRET_KEY", "changeme")
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = int(os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "60"))
DATABASE_URL = os.getenv("DATABASE_URL", "sqlite:///./test.db")  

# --- DB ---
engine = create_engine(DATABASE_URL, connect_args={"check_same_thread": False} if "sqlite" in DATABASE_URL else {})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base() 

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True, index=True)
    email = Column(String, unique=True, index=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    full_name = Column(String, nullable=True)
    is_admin = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)

    preds = relationship("Prediction", back_populates="user")

class Prediction(Base):
    __tablename__ = "predictions"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"))
    input_data = Column(JSON)
    predicted_label = Column(String)
    probabilities = Column(JSON)
    model_version = Column(String)
    created_at = Column(DateTime, default=datetime.utcnow)

    user = relationship("User", back_populates="preds")

Base.metadata.create_all(bind=engine)

# --- Security ---
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/auth/login")

def get_password_hash(password): return pwd_context.hash(password)
def verify_password(plain, hashed): return pwd_context.verify(plain, hashed)

def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

def get_user_by_email(db, email: str):
    return db.query(User).filter(User.email==email).first()

def get_current_user(token: str = Depends(oauth2_scheme), db: SessionLocal = Depends(get_db)):
    credentials_exception = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Could not validate credentials")
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        user_id: int = payload.get("sub")
        if user_id is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    user = db.query(User).filter(User.id==user_id).first()
    if user is None:
        raise credentials_exception
    return user

# --- App & model load ---
app = FastAPI(title="ObesiTrack - Obesity Risk API", version="1.0.0")
MODEL = None
MODEL_INFO = {}

@app.on_event("startup")
def load_model():
    global MODEL, MODEL_INFO
    try:
        MODEL = joblib.load("model.pkl")   # pipeline saved from train.py
        print("✅ Model loaded successfully")
    except Exception as e:
        print(f"⚠️  No model.pkl found or failed to load: {e}")
        MODEL = None
    try:
        with open("model_info.json") as f:
            MODEL_INFO = json.load(f)
        print("✅ Model info loaded")
    except Exception as e:
        print(f"⚠️  No model_info.json found: {e}")
        MODEL_INFO = {}

# --- ROOT ROUTE (This was missing!) ---
@app.get("/", response_class=HTMLResponse)
def root():
    # Read the HTML file you'll create
    try:
        with open("frontend.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "Frontend file not found. Please create frontend.html"
# --- Health Check ---
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow().isoformat(),
        "model_loaded": MODEL is not None,
        "database": "connected"
    }

# --- Schemas ---
class UserCreate(BaseModel):
    email: str
    password: str
    full_name: Optional[str] = None

class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"

class PredictIn(BaseModel):
    features: Dict[str, Any]

class PredictOut(BaseModel):
    prediction: str
    probabilities: Dict[str, float]
    model_version: Optional[str] = None

# --- Auth endpoints ---
@app.post("/auth/register", status_code=201)
def register(payload: UserCreate, db = Depends(get_db)):
    if get_user_by_email(db, payload.email):
        raise HTTPException(400, "Email already registered")
    user = User(email=payload.email, hashed_password=get_password_hash(payload.password), full_name=payload.full_name)
    db.add(user); db.commit(); db.refresh(user)
    return {"id": user.id, "email": user.email, "full_name": user.full_name}

@app.post("/auth/login", response_model=Token)
def login(form_data: OAuth2PasswordRequestForm = Depends(), db = Depends(get_db)):
    # OAuth2PasswordRequestForm -> username field used for email
    user = get_user_by_email(db, form_data.username)
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Incorrect credentials")
    access_token = create_access_token(data={"sub": user.id})
    return {"access_token": access_token, "token_type": "bearer"}

# --- Predict endpoint ---
@app.post("/predict", response_model=PredictOut)
def predict(payload: PredictIn, current_user: User = Depends(get_current_user), db = Depends(get_db)):
    if MODEL is None:
        raise HTTPException(500, "Model not loaded")
    # MODEL is expected to be a sklearn pipeline: call predict and predict_proba
    X_df = payload.features  # dict of features
    # convert to 1-row pandas DataFrame (simple)
    import pandas as pd
    X = pd.DataFrame([X_df])
    proba = MODEL.predict_proba(X)[0]
    classes = MODEL.classes_.tolist()
    prob_dict = {str(c): float(p) for c,p in zip(classes, proba)}
    pred_label = classes[proba.argmax()]
    # save to DB
    rec = Prediction(user_id=current_user.id, input_data=payload.features, predicted_label=str(pred_label),
                     probabilities=prob_dict, model_version=MODEL_INFO.get("version"))
    db.add(rec); db.commit(); db.refresh(rec)
    return {"prediction": str(pred_label), "probabilities": prob_dict, "model_version": MODEL_INFO.get("version")}

# --- History ---
@app.get("/history")
def history(limit:int=50, offset:int=0, current_user: User = Depends(get_current_user), db = Depends(get_db)):
    q = db.query(Prediction).filter(Prediction.user_id==current_user.id).order_by(Prediction.created_at.desc()).limit(limit).offset(offset)
    items = [{
        "id": p.id,
        "input_data": p.input_data,
        "predicted_label": p.predicted_label,
        "probabilities": p.probabilities,
        "created_at": p.created_at.isoformat()
    } for p in q.all()]
    return {"items": items}

# --- Admin: list users ---
@app.get("/admin/users")
def list_users(current_user: User = Depends(get_current_user), db=Depends(get_db)):
    if not current_user.is_admin:
        raise HTTPException(403, "Admin only")
    users = db.query(User).all()
    return [{"id":u.id,"email":u.email,"is_admin":u.is_admin,"created_at":u.created_at.isoformat()} for u in users]

# --- Metrics endpoint ---
@app.get("/metrics")
def metrics():
    return MODEL_INFO