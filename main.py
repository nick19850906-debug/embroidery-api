from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import google.generativeai as genai

app = FastAPI()

# 모든 접속 허용 (CORS)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# API 키 설정
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# 모델 정의 (가장 안정적인 호출 방식)
model = genai.GenerativeModel('gemini-1.5-flash')

def calculate_stitch_count(image_bytes: bytes) -> int:
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
        
        # 이미지 데이터와 프롬프트 결합
        prompt = f"당신은 자수 전문가입니다. 1차 계산된 {estimated_stitches}침을 참고하여 도안 견적서를 한국어로 작성하세요."
        
        response = model.generate_content([
            prompt,
            {'mime_type': file.content_type, 'data': image_bytes}
        ])
        
        return {"expert_quote": response.text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
