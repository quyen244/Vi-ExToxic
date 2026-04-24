import uvicorn
from fastapi import FastAPI, HTTPException 
from pydantic import BaseModel
from model_engine import QwenInferenceService 

# --- 1. ĐỊNH NGHĨA SCHEMA ---
class AnalysisRequest(BaseModel):
    text: str
    emotion: str

class AnalysisResponse(BaseModel):
    reasoning_scaffolding: dict
    thought_trace: str
    final_label: str
    confidence_score: float

# --- 2. KHỞI TẠO APP & SERVICE ---
app = FastAPI(title="Vi-ExToxic Inference Service")

# Khởi tạo instance của Qwen Service (Chỉ load model 1 lần khi start server)
qwen_service = QwenInferenceService()

# --- 3. ENDPOINTS ---
@app.get("/health")
def health_check():
    return {
        "status": "healthy", 
        "model": qwen_service.model_path,
        "device": "cuda"
    }

@app.post("/analyze", response_model=AnalysisResponse)
async def analyze_text(request: AnalysisRequest):
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="Text cannot be empty")

    try:
        # Gọi logic xử lý từ Service
        res_json = await qwen_service.predict_toxic(request.text, request.emotion)
        
        # Trả về kết quả khớp với Schema
        return AnalysisResponse(
            reasoning_scaffolding=res_json.get("reasoning_scaffolding", {}),
            thought_trace=res_json.get("thought_trace", "Analysis failed"),
            final_label=res_json.get("final_label", "Unknown"),
            confidence_score=res_json.get("confidence_score", 0.0)
        )

    except Exception as e:
        print(f"Error during analysis: {e}")
        raise HTTPException(status_code=500, detail="Inference engine error")

# --- 4. RUN SERVER ---
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)