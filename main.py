from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse, FileResponse
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

# --- CORS 설정 (아임웹 연동) ---
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 🚀 기본 설정으로 원상 복구 (새 API 키로 속도 문제 해결됨)
client = genai.Client(
    api_key=os.environ.get("GEMINI_API_KEY")
)
)

def calculate_stitch_count(image_bytes: bytes) -> int:
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError("이미지를 해독할 수 없습니다. AI/PDF 파일이거나 손상된 파일입니다.")

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
def serve_index():
    return {"status": "ok", "message": "Jasu Lab API is running perfectly!"}

@app.post("/api/estimate")
def estimate_embroidery(
    file: UploadFile = File(...),
    width: str = Form("5"),
    quantity: int = Form(50),
    colors: str = Form("4~6도"),
    thread_type: str = Form("일반 레이온사"),
    position: str = Form("좌측 가슴"),
    fabric: str = Form("일반 면/폴리"),
    finishing: str = Form("원단 직접 직조(일반)")
):
    print("👉 [LOG] 1. 아임웹에서 자수 견적 요청 및 세부 옵션 정상 도착!")
    try:
        api_key = os.environ.get("GEMINI_API_KEY")
        if not api_key:
            return JSONResponse(status_code=500, content={"expert_quote": "❌ Render.com에 GEMINI_API_KEY가 설정되지 않았습니다."})

        image_bytes = file.file.read()
        filename = file.filename.lower() if file.filename else ""
        mime_type = file.content_type if file.content_type else "image/png"
        print(f"👉 [LOG] 2. 파일 수신 완료 (파일명: {filename})")

        if filename.endswith(".ai") or filename.endswith(".pdf"):
            if not HAS_FITZ:
                return JSONResponse(status_code=500, content={"expert_quote": "<div style='text-align:center; color:#e74c3c;'><strong>❌ 서버 설정 오류</strong><br>PyMuPDF 누락.</div>"})
            try:
                doc = fitz.open(stream=image_bytes, filetype="pdf")
                page = doc.load_page(0)
                pix = page.get_pixmap(dpi=150)
                image_bytes = pix.tobytes("png")
                mime_type = "image/png"
            except Exception as convert_err:
                return JSONResponse(status_code=422, content={"expert_quote": "<div style='text-align:center; color:#e74c3c;'><strong>❌ AI 파일 변환 실패</strong><br>PDF 호환 옵션을 켜고 저장해주세요.</div>"})

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
        사용자가 업로드한 도안과 [고객 요청 세부 옵션]을 분석하여, 브랜드의 가치를 높이는 세련된 텍스트와 시각적 평가 다이어그램이 포함된 견적서를 작성해주세요.
        
        [고객 요청 세부 옵션]
        - 가로 크기: {width} cm
        - 주문 수량: {quantity} 장
        - 색상 도수: {colors}
        - 실 종류: {thread_type}
        - 자수 위치: {position}
        - 원단 재질: {fabric}
        - 후가공/마감: {finishing}
        - 1차 예상 침수: {estimated_stitches} 침
        
        [단가 계산 지침]
        1. 펀칭비(세팅비): 기본 30,000원. {colors}가 7도 이상이거나 {fabric}이 까다로우면 40,000원으로 상향.
        2. 기본 작업비: 1,000침 당 2,000원 기준.
        3. 실/원단 할증: '{fabric}'이 데님/가죽/신축성이거나 '{thread_type}'이 메탈릭/네온일 경우 작업비에 20% 할증. 3D 입체자수는 30% 할증.
        4. 후가공 비용: '{finishing}'이 열접착, 벨크로, 핀도매일 경우 1장당 500원~1000원의 마감비 추가 부과.
        5. 수량 할인: '{quantity}'장이 50장 이상 30% 할인, 100장 이상 50% 할인. (후가공 비용은 할인 제외)
        6. 최종 단가: 펀칭비 + (할인/할증 적용된 1장당 작업비 × 수량) + (후가공비 × 수량)
        7. ★절대 주의: 모든 금액은 가독성을 위해 반드시 천 단위마다 콤마(,)를 찍으세요.

        [응답 서식 구조 (반드시 아래 HTML 태그 구조를 그대로 복사하여 내용을 채울 것. Markdown 금지, h1~h6 태그 사용 금지)]
        <div class="quote-wrapper">
          <div class="quote-header">
            <div class="quote-h2">자수 도안 분석 및 견적서</div>
            <div class="quote-date">발행일: {today_date}</div>
          </div>
          <div class="quote-body">
            
            <div class="analysis-section">
              <div class="quote-h3">디자인 종합 평가 다이어그램</div>
              <!-- AI가 이미지를 보고 각 항목의 점수(0~100)를 측정하여 아래 style="width: XX%;" 에 반영하세요 -->
              <div class="jasu-eval-grid">
                <div class="jasu-eval-item">
                  <div class="jasu-eval-label">구현 복잡도</div>
                  <div class="jasu-eval-bar-wrap"><div class="jasu-eval-fill" style="width: [점수]%;"></div></div>
                  <div class="jasu-eval-score">[점수]</div>
                </div>
                <div class="jasu-eval-item">
                  <div class="jasu-eval-label">색상 구현력</div>
                  <div class="jasu-eval-bar-wrap"><div class="jasu-eval-fill" style="width: [점수]%;"></div></div>
                  <div class="jasu-eval-score">[점수]</div>
                </div>
                <div class="jasu-eval-item">
                  <div class="jasu-eval-label">타발 밀도</div>
                  <div class="jasu-eval-bar-wrap"><div class="jasu-eval-fill" style="width: [점수]%;"></div></div>
                  <div class="jasu-eval-score">[점수]</div>
                </div>
                <div class="jasu-eval-item">
                  <div class="jasu-eval-label">작업 난이도</div>
                  <div class="jasu-eval-bar-wrap"><div class="jasu-eval-fill" style="width: [점수]%;"></div></div>
                  <div class="jasu-eval-score">[점수]</div>
                </div>
              </div>
              
              <div class="quote-h3">수석 디자이너 코멘트</div>
              <div>[업로드된 도안의 디테일과 고객이 선택한 '{colors}', '{thread_type}', '{finishing}' 옵션이 어떻게 시너지를 내어 명품 퀄리티를 만들어낼 것인지 실무적인 어조로 깊이 있게 해설하세요. 단락이 길면 <br><br> 로 줄바꿈 하세요.]</div>
            </div>

            <div class="table-section">
              <div class="quote-h3">견적 내역 ({quantity}장 기준)</div>
              <table>
                <thead><tr><th>항목</th><th>상세 내용</th><th>금액 (KRW)</th></tr></thead>
                <tbody>
                  <tr><td>초기 세팅 (펀칭)</td><td>패턴 분석, 밀도 세팅 및 디지타이징</td><td>[계산 금액]원</td></tr>
                  <tr><td>자수 가공비</td><td>{estimated_stitches}침 기준 (옵션 할증 및 {quantity}장 할인 적용)</td><td>[계산 금액]원</td></tr>
                  <tr><td>후가공 / 마감</td><td>{finishing} 옵션 비용</td><td>[계산 금액]원</td></tr>
                  <tr class="total-row"><td>총 합계</td><td>(VAT 별도)</td><td>[총 합계 금액]원</td></tr>
                </tbody>
              </table>
            </div>

          </div>
        </div>
        """

        print("👉 [LOG] 3. 구글 제미나이(Gemini)에 디테일 옵션 포함하여 분석 요청 전송 중...")
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type=mime_type)]
        )
        print("👉 [LOG] 4. 분석 및 견적 다이어그램 생성 완료! 프론트엔드로 응답합니다.")
        return {"expert_quote": response.text}

    except Exception as e:
        error_msg = traceback.format_exc()
        print(f"👉 [LOG] 🚨 치명적 에러 발생: {error_msg}")
        return JSONResponse(status_code=500, content={
            "error_detail": str(e),
            "expert_quote": f"❌ 서버 내부 오류 발생: {str(e)}"
        })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
