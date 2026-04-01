from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
import os
import traceback
import uvicorn
from datetime import datetime, timezone, timedelta
from google import genai
from google.genai import types

try:
    import fitz  
    HAS_FITZ = True
except ImportError:
    HAS_FITZ = False

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

    # 거리 변환(Distance Transform)을 이용한 여백(Negative Space) 제외 및 침수 계산 로직
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
    type_pos: str = Form("일반 의류 자수"),
    width: str = Form("10"),
    quantity: int = Form(1),
    fabric: str = Form("일반 면/폴리"),
    punching: str = Form("신규 로고 펀칭"),
    thread: str = Form("일반사 (1~6도)")
):
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return JSONResponse(status_code=500, content={"expert_quote": "❌ Render.com에 GEMINI_API_KEY가 설정되지 않았습니다."})

        image_bytes = file.file.read()
        filename = file.filename.lower() if file.filename else ""
        mime_type = file.content_type if file.content_type else "image/png"

        if filename.endswith(".ai") or filename.endswith(".pdf"):
            if not HAS_FITZ:
                return JSONResponse(status_code=500, content={
                    "expert_quote": "<div style='text-align:center; color:#e74c3c;'><strong>❌ 서버 설정 오류</strong><br>서버에 일러스트 해독기(PyMuPDF)가 설치되지 않았습니다.</div>"
                })
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

        try:
            base_stitches = calculate_stitch_count(image_bytes)
        except ValueError as ve:
            return JSONResponse(status_code=422, content={"expert_quote": f"<div style='text-align:center; color:#e74c3c;'><strong>❌ 해독 불가</strong><br>{str(ve)}</div>"})

        size_ratio = (float(width) / 10.0) ** 2
        estimated_stitches = int(base_stitches * size_ratio)
        if estimated_stitches < 1000:
            estimated_stitches = 1000
            
        KST = timezone(timedelta(hours=9))
        today_date = datetime.now(KST).strftime("%Y-%m-%d")
        
        prompt = f"""
        당신은 20년 이상의 실무 경험을 갖춘 수석 디지털 자수 디자이너이자 B2B 의류 생산 전문가입니다. 
        업로드한 도안과 [고객 요청 옵션]을 바탕으로, 고객에게 깊은 신뢰감을 주는 전문적이고 실무적인 견적서를 작성해주세요.
        
        [고객 요청 옵션]
        - 제작 형태 및 품목: {type_pos}
        - 가로 크기: {width} cm
        - 주문 수량: {quantity} 장
        - 원단 및 대상물: {fabric}
        - 펀칭(도안) 데이터: {punching}
        - 실 종류 및 색상: {thread}
        - 1차 예상 침수: {estimated_stitches} 침
        
        [대한민국 자수 실무 단가 계산 공식 (반드시 엄수할 것)]
        1. 펀칭비(세팅비): '{punching}' 옵션이 '기존 파일 보유'면 0원, '단순 글자/이니셜'이면 11,000원, '신규 로고 펀칭'이면 기본 22,000원에서 복잡도에 따라 상향 적용.
        2. 기본 자수비(1장당): 1,000침 당 2,000원 기준.
        3. 품목 및 후가공 할증(1장당): '{type_pos}'가 '캡모자 자수'면 난이도 할증 부과, 패치의 경우 '열접착'은 +500원, '벨크로'는 +1,500원 추가.
        4. 원단 및 실 할증(1장당): '{fabric}'이 데님/가죽/실크/신축성이거나 '{thread}'가 '7도 이상 다색상' 혹은 '특수사'일 경우 실 교체 시간 및 난이도를 고려하여 자수비의 15~20% 할증 부과.
        5. 수량 할인(도매가): '{quantity}'장이 10장 이상 10%, 30장 이상 20%, 50장 이상 30%, 100장 이상 50%의 자수비 할인 적용.
        6. 최종 총 합계: 펀칭비 + [{{(할인/할증이 적용된 1장당 자수비) + 1장당 후가공비}} * {quantity}].
        7. ★절대 주의(콤마 표기): 모든 금액(특히 총 합계)은 가독성을 위해 반드시 천 단위마다 콤마(,)를 찍으세요. (예: 86,376원)

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
                 <h3>디자인 및 공정 분석</h3>
                 <p>[도안의 형태적 특징, '{type_pos}' 및 '{punching}' 옵션에 따른 공정의 차이, '{fabric}' 원단과 '{thread}'의 조합이 자수 품질에 미치는 영향을 실무적이고 전문적인 어조로 해설하세요. 당사의 'AI 정밀 픽셀 분석 시스템'이 불필요한 여백을 완벽히 배제하고 정확한 바늘땀만 스캔하여 정직한 견적을 산출했음을 어필하세요. ★문단이 길어질 경우 반드시 중간에 <br><br> 태그를 1~2회 삽입하세요.]</p>
               </div>
               <div class="table-section">
                 <h3>견적 내역 ({quantity}장 기준)</h3>
                 <table>
                   <thead><tr><th>항목</th><th>상세 내용</th><th>금액 (KRW)</th></tr></thead>
                   <tbody>
                     <tr><td>초기 세팅비 (펀칭비)</td><td>[{punching} 옵션에 따른 펀칭 작업 내용 명시]</td><td>[계산 금액]원</td></tr>
                     <tr><td>자수 가공비</td><td>[원단, 실 종류 할증 및 수량 할인 적용 내용 명시]</td><td>[계산 금액]원</td></tr>
                     <tr><td>품목/후가공비</td><td>[{type_pos} 적용 내용 명시, 추가금 없으면 0원]</td><td>[계산 금액]원</td></tr>
                     <tr class="total-row"><td>총 합계</td><td>(VAT 별도)</td><td>[최종 합계 금액]원</td></tr>
                   </tbody>
                 </table>
               </div>
             </div>
             <div class="quote-footer">
               <p>※ 본 견적은 AI 정밀 분석에 기반한 가견적이며, 실제 로고 난이도에 따라 단가가 조정될 수 있습니다.</p>
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
