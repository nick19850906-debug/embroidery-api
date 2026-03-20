from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import traceback
import uvicorn
from google import genai
from google.genai import types

app = FastAPI()

# CORS 설정
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

def calculate_stitch_count(image_bytes: bytes) -> int:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None: return 0
    
    # 서버 메모리 초과 방지
    max_width = 600
    height, width = img.shape
    scale_factor = 1.0
    if width > max_width:
        ratio = max_width / width
        new_height = int(height * ratio)
        img = cv2.resize(img, (max_width, new_height), interpolation=cv2.INTER_AREA)
        scale_factor = (width / max_width)**2

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    satin_pixels = np.sum((dist_transform > 0) & (dist_transform < 15))
    tatami_pixels = np.sum(dist_transform >= 15)
    return int(((satin_pixels * 0.15) + (tatami_pixels * 0.25)) * scale_factor)

@app.get("/")
def read_root():
    return {"status": "ok", "message": "API is Live"}

@app.post("/api/estimate")
def estimate_embroidery(file: UploadFile = File(...)):
    try:
        # 1. 여기서 API 키를 안전하게 검사합니다. (서버가 뻗지 않고 화면에 에러를 띄워줍니다)
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return JSONResponse(status_code=500, content={"expert_quote": "❌ Render.com에 GEMINI_API_KEY 환경변수가 설정되지 않았습니다."})
            
        client = genai.Client(api_key=api_key)

        image_bytes = file.file.read()
        estimated_stitches = calculate_stitch_count(image_bytes)
        
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
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[
                prompt,
                types.Part.from_bytes(data=image_bytes, mime_type=file.content_type)
            ]
        )
        
        return {"expert_quote": response.text}
        
    except Exception as e:
        error_msg = traceback.format_exc()
        print(error_msg)
        return JSONResponse(status_code=500, content={"expert_quote": f"❌ 서버 내부 오류 발생: {str(e)}"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
