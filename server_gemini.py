"""
Gemini AI 백엔드 서버 (FastAPI)
- 대상 ngrok URL: https://stem-shifter-selected.ngrok-free.dev
- 실행 방법:
    1) pip install fastapi uvicorn google-genai pydantic
    2) export GEMINI_API_KEY="AIzaSy..." (Windows: set GEMINI_API_KEY=AIzaSy...)
    3) python server_gemini.py
"""

import os
import sys
from typing import Optional
from fastapi import FastAPI, HTTPException, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# FastAPI 앱 생성
app = FastAPI(
    title="Gemini AI Engine for ngrok",
    description="ngrok(https://stem-shifter-selected.ngrok-free.dev) 터널 연동용 Gemini AI 백엔드 엔진",
    version="1.0.0"
)

# 1. CORS 미들웨어 (모든 프론트엔드/브라우저 접속 허용)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. ngrok 무료 계정 안내 화면 자동 바이패스 미들웨어
@app.middleware("http")
async def add_ngrok_bypass_header(request: Request, call_next):
    response = await call_next(request)
    response.headers["ngrok-skip-browser-warning"] = "true"
    return response

# 3. Gemini 클라이언트 초기화
# .env 파일이 로컬에 존재할 경우 자동 로딩
if os.path.exists(".env"):
    try:
        with open(".env", "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line and not line.startswith("#") and "=" in line:
                    k, v = line.split("=", 1)
                    os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
    except Exception:
        pass

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "").strip()

def get_gemini_client():
    key = os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY).strip()
    if not key:
        return None
    try:
        from google import genai
        return genai.Client(api_key=key)
    except ImportError:
        return None

# 요청/응답 데이터 모델
class PromptRequest(BaseModel):
    prompt: str
    model: Optional[str] = "gemini-2.0-flash"  # 또는 "gemini-1.5-flash", "gemini-1.5-pro"
    system_instruction: Optional[str] = "당신은 친절하고 전문적인 AI 어시스턴트입니다."

class PromptResponse(BaseModel):
    status: str
    model: str
    reply: str

@app.get("/")
def health_check():
    key_exists = bool(os.environ.get("GEMINI_API_KEY", GEMINI_API_KEY).strip())
    return {
        "status": "online",
        "service": "Gemini AI ngrok Engine",
        "ngrok_url": "https://stem-shifter-selected.ngrok-free.dev",
        "gemini_api_key_configured": key_exists,
        "guide": "POST /ask 또는 POST /api/chat 엔드포인트로 {\"prompt\": \"질문\"} 형식의 JSON을 전송하세요."
    }

@app.post("/ask", response_model=PromptResponse)
@app.post("/api/chat", response_model=PromptResponse)
def ask_gemini(req: PromptRequest):
    """일반 대화 요청 처리 (단일 응답)"""
    client = get_gemini_client()
    if not client:
        raise HTTPException(
            status_code=500,
            detail="GEMINI_API_KEY 환경변수가 설정되지 않았거나 google-genai 패키지가 설치되지 않았습니다. 'pip install google-genai' 후 키를 설정하세요."
        )

    try:
        response = client.models.generate_content(
            model=req.model,
            contents=req.prompt,
        )
        return PromptResponse(
            status="success",
            model=req.model,
            reply=response.text or "응답이 비어 있습니다."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gemini API 호출 중 오류 발생: {str(e)}")

@app.post("/ask-stream")
def ask_gemini_stream(req: PromptRequest):
    """실시간 타자 효과 스트리밍 응답 (Server-Sent Events)"""
    client = get_gemini_client()
    if not client:
        raise HTTPException(status_code=500, detail="GEMINI_API_KEY가 설정되지 않았습니다.")

    def stream_generator():
        try:
            response_stream = client.models.generate_content_stream(
                model=req.model,
                contents=req.prompt,
            )
            for chunk in response_stream:
                if chunk.text:
                    yield f"data: {chunk.text}\n\n"
            yield "data: [DONE]\n\n"
        except Exception as e:
            yield f"data: [ERROR: {str(e)}]\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    print("=" * 60)
    print("🚀 Gemini AI ngrok 백엔드 서버 구동 시작")
    print("📌 로컬 주소: http://localhost:8000")
    print("📌 ngrok 터널 연결 명령어:")
    print("   ngrok http 8000 --url=stem-shifter-selected.ngrok-free.dev")
    print("=" * 60)
    uvicorn.run("server_gemini:app", host="0.0.0.0", port=8000, reload=True)
