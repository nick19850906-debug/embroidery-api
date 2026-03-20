from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
from google import genai
from google.genai import types

app = FastAPI()

# CORS 설정: 모든 접속 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경 변수 로드
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=GEMINI_API_KEY)

def calculate_stitch_count(image_bytes: bytes) -> int:
    """물리적 침수 계산"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None: return 0
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    satin_pixels = np.sum((dist_transform > 0) & (dist_transform < 15))
    tatami_pixels = np.sum(dist_transform >= 15)
    return int((satin_pixels * 0.15) + (tatami_pixels * 0.25))

@app.get("/")
def read_root():
    return {"status": "ok"}

@app.post("/api/estimate")
async def estimate_embroidery(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        estimated_stitches = calculate_stitch_count(image_bytes)
        
        # 모델 이름을 "gemini-1.5-flash"로 수정 (models/ 제거)
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=[
                "당신은 자수 전문가입니다. 이미지를 분석해 견적을 내주세요.",
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            ]
        )
        
        return {"expert_quote": response.text}
        
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
