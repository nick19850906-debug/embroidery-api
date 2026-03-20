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
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 최신 SDK 클라이언트 설정
# 환경 변수에 GEMINI_API_KEY가 반드시 있어야 합니다.
client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

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
    return {"status": "ok", "message": "API is Live"}

@app.post("/api/estimate")
async def estimate_embroidery(file: UploadFile = File(...)):
    try:
        image_bytes = await file.read()
        estimated_stitches = calculate_stitch_count(image_bytes)
        
        # 최신 모델 호출 방식
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                f"당신은 자수 전문가입니다. 1차 계산된 {estimated_stitches}침을 참고하여 도안 견적서를 한국어로 작성하세요.",
                types.Part.from_bytes(data=image_bytes, mime_type=file.content_type)
            ]
        )
        
        return {"expert_quote": response.text}
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
