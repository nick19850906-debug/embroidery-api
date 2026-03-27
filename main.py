from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import traceback
import uvicorn
import fitz  # ★ AI 및 PDF 해독 라이브러리
from datetime import datetime, timezone, timedelta
from google import genai
from google.genai import types

app = FastAPI()

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
def estimate_embroidery(
    file: UploadFile = File(...),
    width: str = Form("10"),
    quantity: int = Form(1),
    position: str = Form("좌측 가슴"),
    fabric: str = Form("일반 면/폴리")
):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return JSONResponse(status_code=500, content={"expert_quote": "❌ Render.com에 GEMINI_API_KEY가 설정되지 않았습니다."})

        client = genai.Client(api_key=api_key)
        image_bytes = file.file.read()
        filename = file.filename.lower() if file.filename else ""
        mime_type = file.content_type if file.content_type else "image/png"

        # ★ AI 및 PDF 파일 고화질 이미지 변환 (500 에러 해결)
        if filename.endswith(".ai") or filename.endswith(".pdf"):
            try:
                doc = fitz.open(stream=image_bytes, filetype="pdf")
                page = doc.load_page(0)
                pix = page.get_pixmap(dpi=150)
                image_bytes = pix.tobytes("png")
                mime_type = "image/png"
            except Exception as convert_err:
                return JSONResponse(status_code=422, content={
                    "error_detail": str(convert_err),
                    "expert_quote": "<div style='text-align:center; color:#e74c3c;'><strong>❌ AI 파일 변환 실패</strong><br>일러스트레이터에서 저장 시 <b>'PDF 호환 파일 만들기'</b> 옵션을 켜고 저장한 파일만 지원됩니다.</div>"
                })

        base_stitches = calculate_stitch_count(image_bytes)
        size_ratio = (float(width) / 10.0) ** 2
        estimated_stitches = int(base_stitches * size_ratio)
        if estimated_stitches < 1000:
            estimated_stitches = 1000
            
        KST = timezone(timedelta(hours=9))
        today_date = datetime.now(KST).strftime("%Y-%m-%d")
        
        prompt = f"""
        당신은 20년 이상의 실무 경험을 갖춘 수석 디지털 자수 디자이너이자 B2B 의류 생산 전문가입니다. 
        사용자가 업로드한 도안과 [고객 요청 옵션]을 바탕으로, 고객에게 깊은 신뢰감을 주고 브랜드의 가치를 높여줄 수 있는 보편적이면서도 세련된 견적서를 작성해주세요.
        
        [고객 요청 옵션]
        - 가로 크기: {width} cm
        - 주문 수량: {quantity} 장
        - 자수 위치: {position}
        - 원단 재질: {fabric}
        - 1차 예상 침수: {estimated_stitches} 침
        
        [단가 계산 지침]
        1. 펀칭비(세팅비): 기본 30,000원. 복잡하면 상향.
        2. 기본 작업비: 1,000침 당 2,000원 기준.
        3. 원단 할증: '{fabric}'이 데님, 가죽, 실크, 신축성, 3D입체자수일 경우 작업비에 15% 할증 부과. 일반 면/폴리는 할증 없음.
        4. 수량 할인: '{quantity}'장이 50장 이상이면 작업비 30% 할인, 100장 이상이면 50% 할인.
        5. 최종 단가: 펀칭비 + (할인/할증 적용된 1장당 작업비 × 수량)
        6. ★절대 주의: 모든 금액은 가독성을 위해 반드시 천 단위마다 콤마(,)를 찍으세요.

        [응답 서식]
        - 순수 HTML 태그만 출력 (Markdown 금지)
        - 다음 HTML 구조 엄수:
           <div class="quote-wrapper">
             <div class="quote-header">
               <h2>자수 도안 분석 및 견적서</h2>
               <p class="quote-date">발행일: {today_date}</p>
             </div>
             <div class="quote-body">
               <div class="analysis-section">
                 <h3>디자인 및 옵션 분석</h3>
                 <p>[도안의 형태적 특징, 시각적 밸런스, 그리고 선택한 원단과 자수 기법이 어떻게 조화를 이루어 최고 품질의 결과물을 만들어낼 수 있는지 실무적인 어조로 해설하세요. ★문단이 길어질 경우 반드시 중간에 <br><br> 태그를 삽입하여 줄바꿈을 해주세요.]</p>
               </div>
               <div class="table-section">
                 <h3>견적 내역 ({quantity}장 기준)</h3>
                 <table>
                   <thead><tr><th>항목</th><th>상세 내용</th><th>금액 (KRW)</th></tr></thead>
                   <tbody>
                     <tr><td>초기 세팅비 (펀칭비)</td><td>패턴 분석 및 1회성 디지타이징</td><td>[계산 금액]원</td></tr>
                     <tr><td>자수 가공비</td><td>[할증 및 수량 할인 적용 내용 명시]</td><td>[계산 금액]원</td></tr>
                     <tr class="total-row"><td>총 합계</td><td>(VAT 별도)</td><td>[총 합계 금액]원</td></tr>
                   </tbody>
                 </table>
               </div>
             </div>
           </div>
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type=mime_type)]
        )
        return {"expert_quote": response.text}
        
    except Exception as e:
        print(traceback.format_exc())
        return JSONResponse(status_code=500, content={"error_detail": str(e), "expert_quote": f"❌ 서버 내부 오류 발생: {str(e)}"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
