from fastapi import FastAPI
from pydantic import BaseModel
from agent_ollama import build_pipeline

app = FastAPI()
pipeline = build_pipeline()

class InvoiceRequest(BaseModel):
    raw_text: str

@app.post("/process-invoice")
def process_invoice(req: InvoiceRequest):
    result = pipeline.invoke({"raw_text": req.raw_text})
    return {
        "status": result["validation_result"]["status"],
        "reason": result["validation_result"]["reason"],
        "overcharge_usd": result["validation_result"].get("overcharge_usd", 0),
        "escalation_email": result["escalation_email"]
    }