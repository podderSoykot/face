# main.py
from fastapi import FastAPI, UploadFile, File, Form, Depends, HTTPException, Query, Request
from fastapi.responses import HTMLResponse, StreamingResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy import create_engine, Column, Integer, String, LargeBinary, DateTime, ForeignKey
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, relationship, Session
from datetime import datetime
import face_recognition
import pickle
import numpy as np
import cv2
import base64
import io
import csv
import os

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")
templates = Jinja2Templates(directory="templates")

# Database setup
SQLALCHEMY_DATABASE_URL = "sqlite:///./test.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(bind=engine)
Base = declarative_base()

class User(Base):
    __tablename__ = "users"
    id = Column(Integer, primary_key=True)
    name = Column(String, unique=True)
    employee_no = Column(String)
    gender = Column(String)
    birthday = Column(String)
    id_no = Column(String)
    card_no = Column(String)
    email = Column(String)
    position = Column(String)
    department = Column(String)
    start_time = Column(String)
    end_time = Column(String)
    mobile = Column(String)
    company = Column(String)
    remark = Column(String)
    encoding = Column(LargeBinary)
    login_logs = relationship("LoginLog", back_populates="user")

class IPCamera(Base):
    __tablename__ = "cameras"
    id = Column(Integer, primary_key=True)
    name = Column(String)
    ip = Column(String)
    username = Column(String)
    password = Column(String)
    location = Column(String)

class LoginLog(Base):
    __tablename__ = "login_logs"
    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id"), nullable=True)
    name = Column(String, nullable=True)
    location = Column(String)
    time = Column(DateTime, default=datetime.utcnow)
    registered_image = Column(LargeBinary, nullable=True)
    live_image = Column(LargeBinary, nullable=True)
    access_result = Column(String)
    confidence = Column(String)
    user = relationship("User", back_populates="login_logs")

Base.metadata.create_all(bind=engine)

# Dependency

def get_db():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse("home.html", {"request": request})

@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return templates.TemplateResponse("dashboard.html", {"request": request})

@app.get("/register", response_class=HTMLResponse)
def register_page(request: Request):
    return templates.TemplateResponse("register.html", {"request": request})

@app.post("/register/")
async def register_user(
    request: Request,
    name: str = Form(...), employee_no: str = Form(...), gender: str = Form(...),
    birthday: str = Form(...), id_no: str = Form(...), card_no: str = Form(...),
    email: str = Form(...), position: str = Form(...), department: str = Form(...),
    start_time: str = Form(...), end_time: str = Form(...), mobile: str = Form(...),
    company: str = Form(...), remark: str = Form(...), file: UploadFile = File(...),
    db: Session = Depends(get_db)
):
    image_bytes = await file.read()
    image = cv2.imdecode(np.frombuffer(image_bytes, np.uint8), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    if not encodings:
        raise HTTPException(400, "No face detected")
    encoding = pickle.dumps(encodings[0])
    user = User(
        name=name, employee_no=employee_no, gender=gender, birthday=birthday,
        id_no=id_no, card_no=card_no, email=email, position=position,
        department=department, start_time=start_time, end_time=end_time,
        mobile=mobile, company=company, remark=remark, encoding=encoding
    )
    db.add(user)
    db.commit()
    return RedirectResponse(url="/dashboard", status_code=303)

@app.get("/add-device", response_class=HTMLResponse)
def add_device_page(request: Request):
    return templates.TemplateResponse("add-device.html", {"request": request})

@app.post("/add-device/")
async def add_device(
    request: Request,
    name: str = Form(...), ip: str = Form(...), username: str = Form(...),
    password: str = Form(...), location: str = Form(...), db: Session = Depends(get_db)
):
    camera = IPCamera(name=name, ip=ip, username=username, password=password, location=location)
    db.add(camera)
    db.commit()
    return RedirectResponse(url="/dashboard", status_code=303)

@app.get("/login", response_class=HTMLResponse)
def login_page(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@app.post("/login/")
def login(db: Session = Depends(get_db)):
    camera = db.query(IPCamera).first()
    if not camera:
        raise HTTPException(404, "Camera not found")
    stream_url = f"rtsp://{camera.username}:{camera.password}@{camera.ip}:554/Streaming/Channels/101"
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise HTTPException(400, "Failed to open camera")
    ret, frame = cap.read()
    cap.release()
    if not ret:
        raise HTTPException(500, "Failed to read frame")
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    encodings = face_recognition.face_encodings(rgb)
    if not encodings:
        return JSONResponse(status_code=400, content={"detail": "No face detected"})
    live_encoding = encodings[0]
    users = db.query(User).all()
    location = camera.location
    for user in users:
        known_encoding = pickle.loads(user.encoding)
        results = face_recognition.compare_faces([known_encoding], live_encoding)
        distance = face_recognition.face_distance([known_encoding], live_encoding)[0]
        if results[0]:
            _, buffer = cv2.imencode(".jpg", frame)
            log = LoginLog(
                user_id=user.id, name=user.name, location=location,
                registered_image=b"", live_image=buffer.tobytes(),
                access_result="Granted", confidence=f"{1 - distance:.2f}"
            )
            db.add(log)
            db.commit()
            return {"message": f"Welcome, {user.name}", "confidence": f"{1 - distance:.2f}"}
    _, buffer = cv2.imencode(".jpg", frame)
    db.add(LoginLog(name="Unknown", location=location, live_image=buffer.tobytes(), access_result="Denied", confidence="0.00"))
    db.commit()
    return JSONResponse(status_code=401, content={"detail": "Face not recognized"})

@app.get("/live-feed")
def live_feed(db: Session = Depends(get_db)):
    camera = db.query(IPCamera).first()
    if not camera:
        raise HTTPException(status_code=404, detail="No IP camera found.")
    stream_url = f"rtsp://{camera.username}:{camera.password}@{camera.ip}:554/Streaming/Channels/101"
    cap = cv2.VideoCapture(stream_url)
    if not cap.isOpened():
        raise HTTPException(status_code=400, detail="Unable to open IP camera stream.")
    def stream():
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                _, jpeg = cv2.imencode('.jpg', frame)
                yield (b'--frame\r\nContent-Type: image/jpeg\r\n\r\n' + jpeg.tobytes() + b'\r\n')
        finally:
            cap.release()
    return StreamingResponse(stream(), media_type='multipart/x-mixed-replace; boundary=frame')

@app.get("/live-view", response_class=HTMLResponse)
def live_view_page(request: Request):
    return templates.TemplateResponse("live_feed.html", {"request": request})

@app.get("/login-history", response_class=HTMLResponse)
def login_history(request: Request, skip: int = 0, limit: int = 10, db: Session = Depends(get_db)):
    logs = db.query(LoginLog).order_by(LoginLog.time.desc()).offset(skip).limit(limit).all()
    return templates.TemplateResponse("history.html", {"request": request, "logs": logs})
