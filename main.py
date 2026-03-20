from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
from google import genai
from google.genai import types

app = FastAPI()

# 1. CORS 설정: 프론트엔드(CodePen)와의 통신 허용
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. 환경 변수에서 API 키 로드
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

# 3. Gemini 클라이언트 초기화
client = genai.Client(api_key=GEMINI_API_KEY)

def calculate_stitch_count(image_bytes: bytes) -> int:
    """OpenCV를 이용한 물리적 침수 계산"""
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
    return {"status": "ok", "info": "The Jasu Lab API is Ready"}

@app.post("/api/estimate")
async def estimate_embroidery(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        
        # 1차 물리적 계산 실행
        estimated_stitches = calculate_stitch_count(image_bytes)
        
        # 2차 AI 분석 실행 (모델명에서 'models/' 제거하고 순수 이름만 사용)
        response = client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=[
                f"당신은 20년 경력의 자수 전문가입니다. 이 도안의 1차 계산 침수는 {estimated_stitches}침입니다. 이를 바탕으로 상세한 자수 견적서(디자인 분석, 추천 기법, 예상 비용)를 한국어로 작성해주세요.",
                types.Part.from_bytes(data=image_bytes, mime_type=file.content_type)
            ]
        )
        
        return {"expert_quote": response.text}
        
    except Exception as e:
        # 에러 발생 시 로그에 상세 내용 출력
        print(f"CRITICAL ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
