from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import traceback
import uvicorn
from datetime import datetime
from google import genai
from google.genai import types

app = FastAPI()

# CORS 에러 완벽 해결
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
    if img is None: 
        raise ValueError("이미지를 해독할 수 없습니다. 정상적인 이미지인지 확인해주세요.")
    
    # 서버 메모리 초과 방지
    max_dim = 600
    height, width = img.shape
    scale_factor = 1.0
    if max(height, width) > max_dim:
        if width > height:
            new_width = max_dim
            new_height = int(height * (max_dim / width))
        else:
            new_height = max_dim
            new_width = int(width * (max_dim / height))
        img = cv2.resize(img, (new_width, new_height), interpolation=cv2.INTER_AREA)
        scale_factor = (max(height, width) / max_dim) ** 2

    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY_INV)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    satin_pixels = np.sum((dist_transform > 0) & (dist_transform < 15))
    tatami_pixels = np.sum(dist_transform >= 15)
    return int(((satin_pixels * 0.15) + (tatami_pixels * 0.25)) * scale_factor)

@app.get("/")
def read_root():
    return {"status": "awake", "message": "서버가 정상 작동 중입니다."}

@app.post("/api/estimate")
def estimate_embroidery(file: UploadFile = File(...)):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return JSONResponse(status_code=500, content={"expert_quote": "❌ Render.com에 GEMINI_API_KEY 환경변수가 설정되지 않았습니다."})
            
        client = genai.Client(api_key=api_key)
        image_bytes = file.file.read()
        estimated_stitches = calculate_stitch_count(image_bytes)
        today_date = datetime.now().strftime("%Y-%m-%d")
        
        prompt = f"""
        당신은 하라 켄야(Kenya Hara)의 미니멀리즘 철학을 따르는 수석 디지털 자수 디자이너입니다. 
        사용자가 업로드한 도안을 분석하고, 1차 계산된 예상 침수({estimated_stitches}침)를 바탕으로, 최고급 하이엔드 브랜드에 걸맞은 세련된 견적서를 작성해주세요.
        
        [필수 지침]
        1. 모든 가격은 반드시 한국 원화(KRW, 원)로 계산하세요. (초기 펀칭/디지타이징 세팅비: 난이도에 따라 30,000원 ~ 50,000원 / 자수 작업비: 1,000침당 2,000원 기준)
        2. 응답은 순수 HTML 태그만 출력하세요. (```html 등의 마크다운은 절대 금지)
        3. 다음 HTML 구조와 클래스명을 엄격히 지켜 작성하세요:
           <div class="quote-wrapper">
             <div class="quote-header">
               <h2>자수 도안 분석 및 견적서</h2>
               <p class="quote-date">발행일: {today_date}</p>
             </div>
             <div class="quote-body">
               <div class="analysis-section">
                 <h3>디자인 분석</h3>
                 <p>[로고의 형태, 복잡도, 추천 자수 기법(사틴, 다다미 등)을 미니멀하고 전문적인 어조로 서술]</p>
               </div>
               <div class="table-section">
                 <h3>견적 내역</h3>
                 <table>
                   <thead><tr><th>항목</th><th>상세 내용</th><th>금액 (KRW)</th></tr></thead>
                   <tbody>
                     <tr><td>초기 세팅비 (Digitizing)</td><td>패턴 분석 및 펀칭 작업</td><td>[계산 금액]원</td></tr>
                     <tr><td>자수 가공비 (Production)</td><td>예상 침수 {estimated_stitches}침 기준</td><td>[계산 금액]원</td></tr>
                     <tr class="total-row"><td>총 합계</td><td>(VAT 별도)</td><td>[총 합계 금액]원</td></tr>
                   </tbody>
                 </table>
               </div>
             </div>
             <div class="quote-footer">
               <p>※ 본 견적은 AI 정밀 분석에 기반한 가견적이며, 원단과 수량에 따라 최종 단가가 조정될 수 있습니다.</p>
             </div>
           </div>
        """
        
        mime_type = file.content_type if file.content_type else "image/png"
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type=mime_type)]
        )
        
        return {"expert_quote": response.text}
        
    except Exception as e:
        error_msg = traceback.format_exc()
        print(error_msg)
        return JSONResponse(status_code=500, content={"expert_quote": f"❌ 서버 내부 오류 발생: {str(e)}"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
