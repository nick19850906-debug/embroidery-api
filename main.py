import cv2
import numpy as np
import json
import re

# (기존 import문과 FastAPI 설정 유지) ...

def analyze_embroidery_features(image_bytes: bytes):
    """침수, 색상 수, 복잡도를 종합적으로 분석하는 고도화된 함수"""
    nparr = np.frombuffer(image_bytes, np.uint8)
    img_color = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    img_gray = cv2.imdecode(nparr, cv2.IMREAD_GRAYSCALE)
    
    if img_color is None or img_gray is None:
        raise ValueError("이미지를 해독할 수 없습니다.")

    # 1. 크기 정규화
    max_dim = 600
    height, width = img_gray.shape
    scale_factor = 1.0
    if max(height, width) > max_dim:
        if width > height:
            new_width = max_dim
            new_height = int(height * (max_dim / width))
        else:
            new_height = max_dim
            new_width = int(width * (max_dim / height))
        img_gray = cv2.resize(img_gray, (new_width, new_height), interpolation=cv2.INTER_AREA)
        img_color = cv2.resize(img_color, (new_width, new_height), interpolation=cv2.INTER_AREA)
        scale_factor = (max(height, width) / max_dim) ** 2

    # 2. 침수 계산 (기존 로직 유지)
    _, binary = cv2.threshold(img_gray, 127, 255, cv2.THRESH_BINARY_INV)
    dist_transform = cv2.distanceTransform(binary, cv2.DIST_L2, 5)
    satin_pixels = np.sum((dist_transform > 0) & (dist_transform < 15))
    tatami_pixels = np.sum(dist_transform >= 15)
    base_stitches = int(((satin_pixels * 0.15) + (tatami_pixels * 0.25)) * scale_factor)

    # 3. 색상 수 추정 (포스터라이징 후 고유 색상 카운트)
    img_quant = (img_color // 64) * 64 # 색상 단순화
    unique_colors = len(np.unique(img_quant.reshape(-1, 3), axis=0))
    color_count = min(unique_colors, 15) # 최대 15도로 제한

    # 4. 복잡도 계산 (외곽선 개수)
    edges = cv2.Canny(img_gray, 100, 200)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    complexity_score = len(contours)

    return base_stitches, color_count, complexity_score

# ... (API 엔드포인트 부분) ...

@app.post("/api/estimate")
def estimate_embroidery(
    # ... (기존 Form 파라미터 유지) ...
):
    try:
        # ... (API 키 확인 및 PDF/AI 변환 로직 기존과 동일하게 유지) ...

        try:
            base_stitches, color_count, complexity_score = analyze_embroidery_features(image_bytes)
        except ValueError as ve:
            return JSONResponse(status_code=422, content={"expert_quote": f"❌ 해독 불가: {str(ve)}"})

        size_ratio = (float(width) / 10.0) ** 2
        estimated_stitches = max(int(base_stitches * size_ratio), 1000)
        
        # 난이도 텍스트 변환
        difficulty = "평이함"
        if complexity_score > 50: difficulty = "복잡함 (세밀한 펀칭 필요)"
        if complexity_score > 150: difficulty = "매우 복잡함 (고난이도 특수 펀칭 요망)"

        prompt = f"""
        당신은 20년 이상의 실무 경험을 갖춘 수석 디지털 자수 디자이너이자 세무/원가 계산 전문가입니다.
        업로드된 도안과 아래 분석 데이터를 바탕으로 견적을 산출하세요.

        [도안 분석 데이터]
        - 가로 크기: {width} cm
        - 주문 수량: {quantity} 장
        - 자수 위치: {position}
        - 원단 재질: {fabric}
        - 1차 예상 침수: {estimated_stitches} 침
        - 추정 색상 수: 약 {color_count} 도
        - 도안 복잡도: {difficulty}
        
        [단가 계산 지침]
        1. 펀칭비(세팅비): 기본 30,000원. 도안 복잡도가 '복잡함' 이상이거나 색상이 5도를 초과하면 10,000원~20,000원 상향.
        2. 기본 작업비: 1,000침 당 2,000원.
        3. 할증: '{fabric}'이 데님, 가죽, 실크, 신축성, 3D입체자수일 경우 작업비 15% 할증.
        4. 수량 할인: {quantity}장이 50장 이상 시 30%, 100장 이상 시 50% 할인.
        5. 모든 금액 숫자는 콤마(,)를 제외한 순수 정수로만 출력하세요.

        [응답 서식]
        반드시 아래의 엄격한 JSON 형식으로만 응답하세요. 마크다운 기호(```json 등)는 절대 쓰지 마세요.
        {{
            "punching_fee": 30000,
            "unit_price": 2500,
            "total_price": 55000,
            "analysis": "도안의 시각적 밸런스와 원단 조화에 대한 실무적 분석 (줄바꿈은 <br> 태그 사용)",
            "technical_note": "색상 수({color_count}도)와 복잡도({difficulty})가 작업에 미치는 영향 및 고품질 생산을 위한 조언"
        }}
        """
        
        response = client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt, types.Part.from_bytes(data=image_bytes, mime_type=mime_type)]
        )
        
        # Markdown 백틱 제거 및 JSON 파싱
        raw_text = response.text.strip()
        raw_text = re.sub(r'^```json\n', '', raw_text)
        raw_text = re.sub(r'\n```$', '', raw_text)
        
        parsed_data = json.loads(raw_text)
        return {"data": parsed_data}
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"error_detail": str(e), "expert_quote": f"❌ 분석 중 오류 발생: {str(e)}"})
