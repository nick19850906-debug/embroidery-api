<link rel="preconnect" href="https://fonts.googleapis.com">
<link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
<link href="https://fonts.googleapis.com/css2?family=Noto+Serif+KR:wght@400;600&family=Pretendard:wght@300;400;500;600;800&display=swap" rel="stylesheet">

<div class="tjl-dyna-wrapper">
    <canvas id="word-particles"></canvas>
    
    <div class="tjl-dyna-content-wrapper">
        <div class="tjl-dyna-header">
            <div class="tjl-dyna-icon-box">
                <svg class="tjl-dyna-svg-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round">
                    <path d="M14.5 9.5L2 22"></path><path d="M17.5 6.5l-3 3"></path><path d="M22 2l-4.5 4.5"></path>
                    <path d="M18 10c-1.5 1.5-4 4-4 4s-2 1.5-1.5 3.5 2 2 1.5 3-1.5 4-4 4-4 1.5-2.5 1.5-4"></path>
                </svg>
            </div>
            <div class="tjl-dyna-title">AI EMBROIDERY ESTIMATE</div>
            <div class="tjl-dyna-subtitle">정밀한 픽셀 분석, 살아 숨쉬는 자수의 시작</div>
        </div>
        
        <div class="tjl-dyna-upload-area" id="tjl_dyna_dropZone">
            <div class="tjl-dyna-upload-content">
                <div class="tjl-dyna-upload-hoop-icon">
                    <svg viewBox="0 0 100 100" class="tjl-hoop-svg">
                        <circle cx="50" cy="50" r="42" stroke="#f5f5f5" stroke-width="3" fill="none" />
                        <circle cx="50" cy="50" r="36" stroke="#cccccc" stroke-width="2" stroke-dasharray="4 4" fill="none" />
                        <path d="M40 60 L60 40" stroke="#f5f5f5" stroke-width="2.5" stroke-linecap="round"/>
                        <ellipse cx="61.5" cy="38.5" rx="1.5" ry="3" transform="rotate(45 61.5 38.5)" stroke="#f5f5f5" stroke-width="1.5" fill="none"/>
                        <path d="M38 62 Q 25 75 45 80 T 65 70" stroke="#aaaaaa" stroke-width="1.5" fill="none"/>
                    </svg>
                </div>
                <div class="tjl-dyna-upload-text" id="tjl_upload_text">도안을 클릭하거나 이곳으로 드롭하세요</div>
                <div class="tjl-dyna-upload-subtext" id="tjl_upload_subtext">AI, PDF, 일반 이미지 모두 완벽하게 분석합니다</div>
            </div>
            <div class="tjl-dyna-upload-hoop-stitch-animation"></div>
        </div>
        <input type="file" id="tjl_dyna_fileInput" accept="image/*,.ai,.pdf,application/pdf,application/postscript" style="display: none;">
        
        <div class="tjl-dyna-bento-grid">
            <div class="tjl-dyna-bento-item">
                <div class="tjl-dyna-label">제작 형태 및 위치</div>
                <select id="opt_type" class="tjl-dyna-input">
                    <option value="의류 직접 자수 (좌측 가슴)">의류 직접 자수 (좌측 가슴)</option>
                    <option value="의류 직접 자수 (등판/대형)">의류 직접 자수 (등판/대형)</option>
                    <option value="의류 직접 자수 (모자/소매)">의류 직접 자수 (모자/소매)</option>
                    <option value="자수 패치 (열접착 핫멜트)">자수 패치 (열접착 핫멜트 / +500원)</option>
                    <option value="자수 패치 (벨크로 부착)">자수 패치 (벨크로 부착 / +1,500원)</option>
                    <option value="자수 패치 (옷핀 부착)">자수 패치 (옷핀 부착 / +500원)</option>
                </select>
            </div>
            <div class="tjl-dyna-bento-item">
                <div class="tjl-dyna-label">가로 크기 (cm)</div>
                <input type="number" id="opt_width" class="tjl-dyna-input" value="10" min="1" max="50">
            </div>
            <div class="tjl-dyna-bento-item">
                <div class="tjl-dyna-label">주문 수량 (장)</div>
                <input type="number" id="opt_quantity" class="tjl-dyna-input" value="1" min="1">
            </div>
            <div class="tjl-dyna-bento-item">
                <div class="tjl-dyna-label">원단 및 대상물</div>
                <select id="opt_fabric" class="tjl-dyna-input">
                    <option value="일반 면/폴리">일반 면/폴리 (기본)</option>
                    <option value="데님/캔버스">데님/캔버스 (+15% 할증)</option>
                    <option value="가죽">가죽 (+15% 할증)</option>
                    <option value="실크/신축성">실크/신축성 (+15% 할증)</option>
                    <option value="3D 입체자수(폼)">3D 입체자수 (+30% 할증)</option>
                </select>
            </div>
            <div class="tjl-dyna-bento-item">
                <div class="tjl-dyna-label">사용 색상 수</div>
                <select id="opt_colors" class="tjl-dyna-input">
                    <option value="1~5도 (기본)">1~5도 (기본)</option>
                    <option value="6~9도 다색상">6~9도 다색상 (+10% 할증)</option>
                    <option value="10도 이상 풀컬러">10도 이상 풀컬러 (+20% 할증)</option>
                    <option value="특수사 (금/은/네온)">특수사 (금/은/네온 등 / +20% 할증)</option>
                </select>
            </div>
            <div class="tjl-dyna-bento-item">
                <div class="tjl-dyna-label">제작/납기 일정</div>
                <select id="opt_turnaround" class="tjl-dyna-input">
                    <option value="일반 제작 (7~14일)">일반 제작 (7~14일)</option>
                    <option value="빠른 제작 (3~5일)">빠른 제작 (3~5일 / +20% 급행료)</option>
                    <option value="긴급 제작 (24~48시간)">긴급 제작 (24~48시간 / +50% 급행료)</option>
                </select>
            </div>
        </div>
        
        <button type="button" id="tjl_dyna_btn" class="tjl-dyna-btn" disabled style="background-color: #999;">
            <span class="tjl-dyna-btn-text">서버 활성화 중... (최대 1~2분 소요)</span>
        </button>
        
        <div id="tjl_dyna_loader" class="tjl-dyna-loader">
            <div class="tjl-dyna-loader-text">AI가 도안의 질감과 옵션 데이터를 정밀 계산 중입니다...</div>
            <div class="tjl-dyna-scan-line-container">
                <div class="tjl-dyna-scan-line"></div>
            </div>
        </div>
        
        <div id="tjl_dyna_result" class="tjl-dyna-result"></div>
    </div>
</div>

<style>
/* ========================================================
   TJL DYNAMIC STYLE (아임웹 완벽 호환 디자인)
======================================================== */
.tjl-dyna-wrapper { width: 100%; box-sizing: border-box; padding: 60px 0; background-color: #1a1a1a; position: relative; overflow: hidden; color: #f5f5f5; }
#word-particles { position: absolute; top: 0; left: 0; width: 100%; height: 100%; z-index: 0; pointer-events: none; }
.tjl-dyna-content-wrapper { width: 70%; max-width: 1200px; margin: 0 auto; position: relative; z-index: 1; animation: tjlDynaFadeInUp 0.8s cubic-bezier(0.2, 0.8, 0.2, 1) forwards; }
.tjl-dyna-header { display: flex; flex-direction: column; align-items: center; margin-bottom: 40px; }
.tjl-dyna-icon-box { width: 64px; height: 64px; background: linear-gradient(135deg, #111, #444); border-radius: 50%; display: flex; justify-content: center; align-items: center; margin-bottom: 20px; box-shadow: 0 10px 20px rgba(0,0,0,0.3); animation: tjlDynaFloatIcon 3s ease-in-out infinite; transition: transform 0.3s ease; }
.tjl-dyna-wrapper:hover.tjl-dyna-icon-box { animation-play-state: paused; transform: scale(1.15) rotate(15deg); }
.tjl-dyna-svg-icon { width: 30px; height: 30px; color: #fff; }
.tjl-dyna-title { font-size: 26px; font-weight: 800; letter-spacing: -0.5px; margin-bottom: 8px; text-align: center; display: inline-block; background-image: linear-gradient(90deg, #ffffff, #aaaaaa, #f5f5f5, #eaeaea, #ffffff); background-size: 300% auto; -webkit-background-clip: text; background-clip: text; color: transparent!important; -webkit-text-fill-color: transparent!important; animation: tjlGradientShift 5s linear infinite, tjlTitleHeartbeat 2.5s ease-in-out infinite; transform-origin: center; }
.tjl-dyna-subtitle { font-size: 15px; color: #aaa; font-weight: 400; text-align: center; }

.tjl-dyna-upload-area { width: 100%; box-sizing: border-box; padding: 50px 20px; background: rgba(44, 44, 44, 0.7); border: 2px dashed #444; border-radius: 16px; text-align: center; cursor: pointer; position: relative; overflow: hidden; backdrop-filter: blur(5px); transition: all 0.3s cubic-bezier(0.25, 0.8, 0.25, 1); margin-bottom: 30px; }
.tjl-dyna-upload-area:active { transform: scale(0.98); }
.tjl-dyna-upload-area:hover { border-color: #f5f5f5; background: rgba(44, 44, 44, 0.95); box-shadow: 0 15px 30px rgba(0,0,0,0.2); }
.tjl-dyna-upload-area.tjl-dragover { background: #2a3b4c; border-color: #0066ff; transform: scale(1.02); }
.tjl-dyna-upload-hoop-icon { width: 80px; height: 80px; display: flex; justify-content: center; align-items: center; margin: 0 auto 15px; animation: tjlDynaFloat 2.5s ease-in-out infinite; }
.tjl-hoop-svg { width: 100%; height: 100%; }
.tjl-dyna-upload-text { font-size: 16px; font-weight: 700; color: #f5f5f5; margin-bottom: 6px; }
.tjl-dyna-upload-subtext { font-size: 13px; color: #aaa; }

.tjl-dyna-bento-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 15px; margin-bottom: 30px; }
.tjl-dyna-bento-item { background: #2c2c2c; padding: 15px 20px; border-radius: 12px; border: 1px solid #333; transition: all 0.3s ease; text-align: left; }
.tjl-dyna-label { font-size: 12px; font-weight: 700; color: #aaa; margin-bottom: 8px; display: block; }
.tjl-dyna-input { width: 100%; box-sizing: border-box; padding: 12px 14px; background: #333; border: 2px solid transparent; border-radius: 8px; font-size: 14px; color: #f5f5f5; outline: none; transition: all 0.3s ease; }
.tjl-dyna-input:focus { background: #3a3a3a; border-color: #f5f5f5; }

.tjl-dyna-btn { width: 100%; padding: 20px; border: none; border-radius: 12px; background: linear-gradient(270deg, #ffffff, #eaeaea, #ffffff, #dcdcdc); background-size: 300% 300%; color: #1a1a1a; font-size: 16px; font-weight: 800; letter-spacing: 1px; cursor: pointer; box-shadow: 0 10px 20px rgba(0,0,0,0.3); transition: all 0.2s ease; }
.tjl-dyna-btn:hover:not(:disabled) { transform: translateY(-3px); box-shadow: 0 15px 25px rgba(0,0,0,0.4); filter: brightness(1.1); }
.tjl-dyna-btn:disabled { background: #444; color: #aaa; cursor: wait; animation: none; box-shadow: none; }

.tjl-dyna-loader { display: none; text-align: center; margin-top: 40px; }
.tjl-dyna-loader-text { font-size: 14px; font-weight: 600; color: #f5f5f5; margin-bottom: 15px; animation: tjlDynaPulse 1.5s infinite; }
.tjl-dyna-scan-line-container { width: 100%; height: 4px; background: #333; border-radius: 4px; overflow: hidden; position: relative; }
.tjl-dyna-scan-line { position: absolute; top: 0; left: 0; height: 100%; width: 30%; background: linear-gradient(90deg, transparent, #f5f5f5, transparent); animation: tjlDynaScan 1.2s infinite alternate; }

.tjl-dyna-result { margin-top: 40px; display: none; animation: tjlDynaFadeInUp 0.6s ease forwards; }
.tjl-dyna-result.quote-wrapper { background: #2c2c2c; padding: 30px; border-radius: 16px; border: 1px solid #333; box-shadow: 0 20px 40px rgba(0,0,0,0.2); }
.tjl-dyna-result.quote-header { border-bottom: 2px solid #f5f5f5; padding-bottom: 20px; margin-bottom: 25px; }
.tjl-dyna-result h2 { font-size: 20px; font-weight: 800; color: #f5f5f5!important; margin: 0 0 8px 0; background: none; -webkit-text-fill-color: initial!important; } 
.tjl-dyna-result h3 { font-size: 16px; font-weight: 700; color: #eaeaea; margin: 0 0 15px 0; }
.tjl-dyna-result.quote-date { font-size: 12px; color: #aaa; margin: 0; }
.tjl-dyna-result.analysis-section p { font-size: 14px; line-height: 1.8; color: #ccc; font-weight: 300; margin: 0; word-break: keep-all; }
.tjl-dyna-result.table-section table { width: 100%; border-collapse: collapse; font-size: 14px; line-height: 1.4; margin-top: 20px; }
.tjl-dyna-result.table-section th { border-bottom: 1px solid #444; padding: 16px 0; text-align: left; color: #aaa; font-weight: 600; }
.tjl-dyna-result.table-section td { border-bottom: 1px dashed #333; padding: 16px 0; color: #f5f5f5; }
.tjl-dyna-result.table-section tr:hover td { background-color: #333; padding-left: 5px; }
.tjl-dyna-result.table-section td:last-child,.tjl-dyna-result.table-section th:last-child { text-align: right; }
.tjl-dyna-result.table-section tr.total-row td { border-bottom: none; border-top: 2px solid #f5f5f5; font-weight: 800; font-size: 18px; padding-top: 24px; color: #f5f5f5; }
.tjl-dyna-result.quote-footer { margin-top: 30px; text-align: center; border-top: 1px dashed #444; padding-top: 20px; }
.tjl-dyna-result.quote-footer p { font-size: 11px; color: #777; margin: 0; }

@keyframes tjlDynaFadeInUp { from { opacity: 0; transform: translateY(40px); } to { opacity: 1; transform: translateY(0); } }
@keyframes tjlDynaFadeIn { from { opacity: 0; } to { opacity: 1; } }
@keyframes tjlDynaFloatIcon { 0%, 100% { transform: translateY(0px); } 50% { transform: translateY(-8px); } }
@keyframes tjlDynaFloat { 0%, 100% { transform: translateY(0px); } 50% { transform: translateY(-6px); } }
@keyframes tjlDynaRotateStitch { from { transform: rotate(0deg); } to { transform: rotate(360deg); } }
@keyframes tjlDynaGradientMove { 0% { background-position: 0% 50%; } 50% { background-position: 100% 50%; } 100% { background-position: 0% 50%; } }
@keyframes tjlDynaPulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.4; } }
@keyframes tjlDynaScan { 0% { left: -30%; } 100% { left: 100%; } }
@keyframes tjlGradientShift { 0% { background-position: 100% center; } 100% { background-position: 0% center; } }
@keyframes tjlTitleHeartbeat { 0%, 100%, 20%, 45% { transform: scale(1); text-shadow: 0 0 5px rgba(255,255,255,0); } 10% { transform: scale(1.08); text-shadow: 0 0 15px rgba(255,255,255,0.4); } 30% { transform: scale(1.12); text-shadow: 0 0 20px rgba(255,255,255,0.6); } }

/* 모바일 1열 반응형 완벽 보장 */
@media screen and (max-width: 768px) {
  .tjl-dyna-content-wrapper { width: 100%; padding: 0 15px; }
  .tjl-dyna-bento-grid { grid-template-columns: 1fr; gap: 12px; }
  .tjl-dyna-upload-area { padding: 40px 15px; }
  .tjl-dyna-title { font-size: 22px; } 
  .tjl-dyna-result.table-section table { font-size: 12px; }
  .cursor, #cursor,.custom-cursor,.mouse-cursor { display: none!important; opacity: 0!important; visibility: hidden!important; }
}
</style>

<script>
// 아임웹의 지연 렌더링에 대응하기 위한 "강제 바인딩 폴링(Polling)" 스크립트입니다.
(function() {
    var SERVER_URL = 'https://embroidery-api-zj1s.onrender.com';
    var selectedFile = null; 
    
    // 캔버스 배경 파티클 효과 함수
    function initParticles() {
        var canvas = document.getElementById('word-particles');
        if (!canvas) return;
        var ctx = canvas.getContext('2d');
        var particles =;
        var words =;
        
        function resize() {
            canvas.width = canvas.parentElement.offsetWidth;
            canvas.height = canvas.parentElement.offsetHeight;
        }
        window.addEventListener('resize', resize);
        resize();

        for (var i = 0; i < 40; i++) {
            particles.push({
                x: Math.random() * canvas.width, y: Math.random() * canvas.height,
                vx: (Math.random() - 0.5) * 0.8, vy: (Math.random() - 0.5) * 0.8,
                word: words[Math.floor(Math.random() * words.length)],
                opacity: Math.random() * 0.3 + 0.05, size: Math.random() * 14 + 10 
            });
        }

        function animate() {
            ctx.clearRect(0, 0, canvas.width, canvas.height);
            for (var i = 0; i < particles.length; i++) {
                var p = particles[i];
                p.x += p.vx; p.y += p.vy;
                if (p.x < 0 |

| p.x > canvas.width) p.vx *= -1;
                if (p.y < 0 |

| p.y > canvas.height) p.vy *= -1;
                ctx.globalAlpha = p.opacity; 
                ctx.fillStyle = '#f5f5f5'; 
                ctx.font = p.size + 'px Pretendard, sans-serif';
                ctx.fillText(p.word, p.x, p.y);
            }
            ctx.globalAlpha = 1;
            requestAnimationFrame(animate);
        }
        animate();
    }

    // 아임웹 코드 위젯 요소가 생성될 때까지 0.5초마다 찾습니다.
    var initTimer = setInterval(function() {
        var dropZone = document.getElementById('tjl_dyna_dropZone');
        var fileInput = document.getElementById('tjl_dyna_fileInput');
        var estimateBtn = document.getElementById('tjl_dyna_btn');
        var resultDiv = document.getElementById('tjl_dyna_result');
        var loaderDiv = document.getElementById('tjl_dyna_loader');

        // 요소가 하나라도 없으면 다음 타이머를 기다립니다.
        if (!dropZone ||!fileInput ||!estimateBtn) return;
        
        // 요소가 찾아졌으면 타이머를 멈추고 기능 주입을 시작합니다.
        clearInterval(initTimer);

        // 중복 실행 방지
        if (dropZone.dataset.initialized) return;
        dropZone.dataset.initialized = "true";

        initParticles();

        // 서버 깨우기 핑
        function pingServer() {
            fetch(SERVER_URL + '/', { method: 'GET', mode: 'cors' })
          .then(function(res) {
                if (res.ok) {
                    estimateBtn.disabled = false;
                    estimateBtn.querySelector('.tjl-dyna-btn-text').innerText = "AI 견적 산출 시작하기";
                    estimateBtn.style.animation = "tjlDynaGradientMove 4s ease infinite";
                } else {
                    setTimeout(pingServer, 5000);
                }
            }).catch(function() {
                setTimeout(pingServer, 5000);
            });
        }
        pingServer();

        // UI 파일명 표시 함수
        function updateUI(file) {
            selectedFile = file; // 금고에 저장
            document.getElementById('tjl_upload_text').innerHTML = "<span style='color:#f5f5f5; font-size:18px;'>✓ " + file.name + "</span>";
            document.getElementById('tjl_upload_subtext').innerText = "스캔이 준비되었습니다.";
            dropZone.style.borderStyle = 'solid';
            dropZone.style.borderColor = '#f5f5f5';
        }

        // 이벤트 연결: 인라인 스크립트를 사용하지 않고 직접 이벤트 리스너를 붙입니다.
        dropZone.addEventListener('click', function(e) {
            e.preventDefault();
            fileInput.click();
        });

        fileInput.addEventListener('change', function(e) {
            if (e.target.files && e.target.files.length > 0) {
                updateUI(e.target.files); // 단일 파일 추출
            }
        });

        // 드래그 앤 드롭 완벽 방어 (아임웹 자체 이벤트 가로채기 차단)
        dropZone.addEventListener('dragover', function(e) {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.add('tjl-dragover');
        });

        dropZone.addEventListener('dragleave', function(e) {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('tjl-dragover');
        });

        dropZone.addEventListener('drop', function(e) {
            e.preventDefault();
            e.stopPropagation();
            dropZone.classList.remove('tjl-dragover');
            
            if (e.dataTransfer.files && e.dataTransfer.files.length > 0) {
                fileInput.files = e.dataTransfer.files;
                updateUI(e.dataTransfer.files); // 단일 파일 추출
            }
        });

        // 메인 견적 전송 함수
        estimateBtn.addEventListener('click', async function(e) {
            e.preventDefault();

            if (!selectedFile) {
                alert('도안 파일(이미지, PDF, AI)을 먼저 업로드해주세요.');
                return;
            }

            loaderDiv.style.display = 'block';
            resultDiv.style.display = 'none';
            resultDiv.innerHTML = '';
            estimateBtn.disabled = true;
            estimateBtn.querySelector('.tjl-dyna-btn-text').innerText = "분석 진행 중... (무료 서버 기상으로 최대 1분 소요)";
            estimateBtn.style.animation = "none";
            estimateBtn.style.background = "#444";

            var formData = new FormData();
            formData.append('file', selectedFile); 
            
            var tObj = document.getElementById('opt_type');
            var wObj = document.getElementById('opt_width');
            var qObj = document.getElementById('opt_quantity');
            var fObj = document.getElementById('opt_fabric');
            var cObj = document.getElementById('opt_colors');
            var turnObj = document.getElementById('opt_turnaround');

            formData.append('type_pos', tObj? tObj.value : "의류 직접 자수 (좌측 가슴)");
            formData.append('width', wObj? wObj.value : "10");
            formData.append('quantity', qObj? qObj.value : "1");
            formData.append('fabric', fObj? fObj.value : "일반 면/폴리");
            formData.append('colors', cObj? cObj.value : "1~5도 (기본)");
            formData.append('turnaround', turnObj? turnObj.value : "일반 제작 (7~14일)");

            try {
                var response = await fetch(SERVER_URL + '/api/estimate', {
                    method: 'POST',
                    body: formData
                });

                var data = null;
                try {
                    data = await response.json();
                } catch (jsonErr) {
                    throw new Error("서버가 기지개를 켜고 있습니다. Render.com 무료 서버 특성상 첫 호출 시 약 50초 정도 소요되니, 1분 뒤에 한 번 더 버튼을 눌러주세요!");
                }

                if (!response.ok) {
                    var errMsg = "에러 코드: " + response.status + "<br><br>";
                    if (data.detail) errMsg += "상세 원인: " + JSON.stringify(data.detail);
                    else if (data.expert_quote) errMsg += data.expert_quote;
                    else errMsg += "알 수 없는 에러 발생";
                    throw new Error(errMsg);
                }

                // 성공적으로 데이터를 받으면 화면 출력
                resultDiv.innerHTML = data.expert_quote;
                resultDiv.style.display = 'block';
                setTimeout(() => { resultDiv.scrollIntoView({ behavior: 'smooth', block: 'start' }); }, 100);

            } catch (error) {
                resultDiv.style.display = 'block';
                resultDiv.innerHTML = `
                    <div style="text-align:center; padding:30px; border:2px solid #ff4444; border-radius:12px; background:#2c2c2c;">
                        <div style="font-size:24px; margin-bottom:10px; animation: tjlDynaPulse 1s infinite; color:#ff4444;">⚠️</div>
                        <strong style="color:#ff4444; font-size:16px;">통신 오류 발생</strong><br><br>
                        <span style="color:#aaa; font-size:14px;">${error.message}</span>
                    </div>`;
            } finally {
                loaderDiv.style.display = 'none';
                estimateBtn.disabled = false;
                estimateBtn.querySelector('.tjl-dyna-btn-text').innerText = "AI 견적 산출 시작하기";
                estimateBtn.style.animation = "tjlDynaGradientMove 4s ease infinite";
                estimateBtn.style.background = "linear-gradient(270deg, #ffffff, #eaeaea, #ffffff, #dcdcdc)";
            }
        });

    }, 500); // 즉시 실행 함수 종료
})(); 
</script>
