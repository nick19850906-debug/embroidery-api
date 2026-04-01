from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import traceback
import uvicorn
import fitz  # AI 및 PDF 변환 라이브러리
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

client = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

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
    fabric: str = Form("일반 면/폴리"),
    colors: str = Form("1~6도 (기본)"),
    turnaround: str = Form("일반 제작")
):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return JSONResponse(status_code=500, content={"expert_quote": "❌ Render.com에 GEMINI_API_KEY가 설정되지 않았습니다."})

        image_bytes = file.file.read()
        filename = file.filename.lower() if file.filename else ""
        mime_type = file.content_type if file.content_type else "image/png"

        # AI 파일 및 PDF 고화질 변환
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
        
        # ★ 수정된 프롬프트: 지엽적인 예술 철학을 제외하고, AI 시스템의 정밀도와 고객 맞춤형 해설을 강화했습니다.
        prompt = f"""
        당신은 20년 이상의 실무 경험을 갖춘 수석 디지털 자수 디자이너이자 B2B 의류 생산 전문가입니다. 
        업로드한 도안과 [고객 요청 옵션]을 바탕으로, 누구나 쉽게 이해할 수 있으면서도 당사의 기술력에 깊은 신뢰감을 느낄 수 있는 세련된 견적서를 작성해주세요.
        
        [고객 요청 옵션]
        - 가로 크기: {width} cm
        - 주문 수량: {quantity} 장
        - 자수 위치: {position}
        - 원단 재질: {fabric}
        - 사용 색상: {colors}
        - 납기 일정: {turnaround}
        - 1차 예상 침수: {estimated_stitches} 침
        
        [실무 단가 계산 지침]
        1. 펀칭비(세팅비): 기본 30,000원. 복잡도에 따라 상향.
        2. 기본 작업비: 1,000침 당 2,000원 기준.
        3. 원단/색상 할증: '{fabric}'이 특수 원단이거나, '{colors}'가 '7도 이상' 또는 '특수사'일 경우 작업비에 각각 15% 할증 부과.
        4. 수량 할인(도매가): '{quantity}'장이 12장 이상이면 15%, 48장 이상이면 30%, 100장 이상이면 50%의 작업비 할인 적용.
        5. 납기일 할증(Rush Fee): '{turnaround}'이 '빠른 제작'이면 총액의 25% 할증, '긴급 제작'이면 총액의 50% 할증 적용.
        6. ★절대 주의(콤마 표기): 모든 금액(특히 총 합계)은 가독성을 위해 반드시 천 단위마다 콤마(,)를 찍으세요. (예: 50000 -> 50,000)

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
                 <h3>AI 디자인 및 공정 분석</h3>
                 <p>[업로드된 도안의 특징을 친절하게 설명하고, 선택된 옵션들이 실제 자수 공정 및 품질에 미치는 영향을 설명하세요. 또한, 당사의 'AI 정밀 픽셀 분석 시스템'이 불필요한 여백을 배제하고 오직 실이 박히는 부분만 스캔하여 거품 없고 정직한 맞춤형 견적을 산출했음을 자연스럽게 어필하여 고객의 신뢰를 이끌어내세요. ★문단이 길어질 경우 반드시 중간에 <br><br> 태그를 삽입하세요.]</p>
               </div>
               <div class="table-section">
                 <h3>견적 내역 ({quantity}장 기준)</h3>
                 <table>
                   <thead><tr><th>항목</th><th>상세 내용</th><th>금액 (KRW)</th></tr></thead>
                   <tbody>
                     <tr><td>초기 세팅비 (펀칭비)</td><td>패턴 분석 및 1회성 디지타이징</td><td>[계산 금액]원</td></tr>
                     <tr><td>자수 가공비</td><td>[할증 및 수량 할인 적용 내용 명시]</td><td>[계산 금액]원</td></tr>
                     <tr><td>급행 수수료 (Rush Fee)</td><td>[납기 일정에 따른 할증액 명시, 없으면 0원]</td><td>[계산 금액]원</td></tr>
                     <tr class="total-row"><td>총 합계</td><td>(VAT 별도)</td><td>[최종 합계 금액]원</td></tr>
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
