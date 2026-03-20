from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import google.generativeai as genai  # 라이브러리 호출 방식 변경

app = FastAPI()

# 1. CORS 설정: 모든 접속 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. API 키 설정 (Render 환경변수 GEMINI_API_KEY 사용)
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
genai.configure(api_key=GEMINI_API_KEY)

# 3. 모델 설정 (가장 안정적인 방식)
model = genai.GenerativeModel('gemini-1.5-flash')

def calculate_stitch_count(image_bytes: bytes) -> int:
    """OpenCV 물리적 침수 계산"""
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
    return {"status": "ok", "message": "The Jasu Lab API is Ready"}

@app.post("/api/estimate")
async def estimate_embroidery(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        estimated_stitches = calculate_stitch_count(image_bytes)
        
        # 4. AI 분석 수행 (가장 안정적인 이미지 전달 방식)
        prompt = f"당신은 자수 전문가입니다. 1차 계산된 {estimated_stitches}침을 참고하여 이 도안의 견적서(디자인 분석, 추천 기법, 예상 비용)를 한국어로 작성해주세요."
        
        # 이미지 전처리 및 전송
        contents = [
            prompt,
            {'mime_type': file.content_type, 'data': image_bytes}
        ]
        
        response = model.generate_content(contents)
        
        return {"expert_quote": response.text}
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
