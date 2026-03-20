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
        
# 세련된 HTML 견적서 출력을 위한 프롬프트 정의
        prompt = f"""
        당신은 20년 경력의 수석 디지털 자수 전문가입니다. 
        사용자가 업로드한 자수 도안 이미지를 분석하고, 1차 계산된 예상 침수({estimated_stitches}침)를 바탕으로 세련되고 전문적인 견적서를 작성해주세요.
        
        [필수 포함 및 서식 지침]
        1. 응답은 반드시 HTML 태그(<h2>, <table>, <ul> 등)를 사용하여 작성하세요. Markdown 기호(```html 등)는 절대 포함하지 마세요.
        2. 디자인 분석: 로고의 복잡도, 추천 자수 기법(사틴, 다다미 등)을 전문가의 어조로 서술하세요.
        3. 견적 내역 (HTML <table> 사용):
           - 초기 펀칭(디지타이징) 세팅비: 난이도에 따라 $20 ~ $50 사이로 합리적으로 산정
           - 자수 작업비: 1,000침당 $1.50 기준으로 {estimated_stitches}침 계산
           - 합계 금액 (Total)
        4. 하단에 작은 글씨(<small> 태그)로 "※ 본 견적은 AI 분석에 기반한 가견적이며, 실제 원단 재질 및 주문 수량에 따라 최종 단가가 변동될 수 있습니다."라는 안내 문구를 추가하세요.
        """

        # 최신 모델 호출 방식
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=file.content_type)
            ]
        )
        
        return {"expert_quote": response.text}
        
    except Exception as e:
        print(f"ERROR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
