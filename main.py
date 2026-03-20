from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
from google import genai
from google.genai import types

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], # 실제 서비스 시에는 아임웹/카페24 주소로 변경 권장
    allow_methods=["*"],
    allow_headers=["*"],
)

# 환경 변수에서 API 키 로드
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise ValueError("GEMINI_API_KEY 환경 변수가 설정되지 않았습니다.")

client = genai.Client(api_key=GEMINI_API_KEY)

def calculate_stitch_count(image_bytes: bytes) -> int:
    """OpenCV를 이용한 물리적 침수 추정"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return 0
    
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    
    satin_pixels = np.sum((dist_transform > 0) & (dist_transform < 15))
    tatami_pixels = np.sum(dist_transform >= 15)
    
    return int((satin_pixels * 0.15) + (tatami_pixels * 0.25))

@app.get("/")
def read_root():
    return {"status": "ok", "message": "The Jasu Lab API is running"}

@app.post("/api/estimate")
async def estimate_embroidery(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        
        # 1. 물리적 침수 계산
        estimated_stitches = calculate_stitch_count(image_bytes)
        
        # 2. Gemini API 분석
        prompt = f"""
        당신은 20년 경력의 디지털 자수 전문가입니다. 
        사용자가 업로드한 자수 도안 이미지를 분석하고, 시스템이 1차적으로 계산한 예상 침수({estimated_stitches}침)를 바탕으로 최종 견적을 산출하세요.
        
        결과에는 다음이 포함되어야 합니다:
        1. 디자인 복잡도 평가 (상/중/하)
        2. 예상 총 침수 (오차범위 포함)
        3. 추천 작업 방식 (새틴/다다미 등)
        4. 예상 제작 시간
        
        모든 설명은 친절한 한국어로 답변하세요.
        """
        
        response = client.models.generate_content(
            model="gemini-2.0-flash", # 또는 사용 가능한 최신 모델
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type="image/jpeg")
            ]
        )
        
        return {"expert_quote": response.text}
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
