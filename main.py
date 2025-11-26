# main.py
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware # [1] 이 줄 추가 (필수!)
import shutil
import os
from inference import HighlightDetector

app = FastAPI()

origins = [
    "http://localhost:3000",              # 로컬 개발 환경
    "https://video-transform.vercel.app", # 배포된 프론트엔드 주소
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,      # 허용할 출처 목록
    allow_credentials=True,     # 쿠키 등 인증 정보 허용 여부
    allow_methods=["*"],        # 허용할 HTTP 메서드 (GET, POST, PUT 등 전체 허용)
    allow_headers=["*"],        # 허용할 헤더 (전체 허용)
)

# 1. 서버 시작 시 모델 로드 (한 번만 로드하여 메모리에 상주)
# 실제 경로에 맞게 수정 필요
MODEL_PATH = "./models/mlp.pt"
SCALER_PATH = "./models/scaler.pkl"

detector = None

@app.on_event("startup")
async def startup_event():
    global detector
    if os.path.exists(MODEL_PATH) and os.path.exists(SCALER_PATH):
        detector = HighlightDetector(MODEL_PATH, SCALER_PATH)
        print("✅ Highlight Detector Initialized")
    else:
        print("❌ Model files not found!")

# 2. 추론 엔드포인트
@app.post("/detect-highlights")
async def detect_highlights(file: UploadFile = File(...)):
    if not detector:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    # 임시 파일 저장
    temp_filename = f"temp_{file.filename}"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)
        
    try:
        # 추론 실행
        highlights = detector.predict(temp_filename)
        
        # 결과 반환 (JSON)
        return {
            "filename": file.filename,
            "highlights": highlights  # [{"start": 10.5, "end": 30.5}, ...]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
        
    finally:
        # 임시 파일 삭제
        if os.path.exists(temp_filename):
            os.remove(temp_filename)