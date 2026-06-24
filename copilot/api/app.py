from fastapi import FastAPI


app = FastAPI(
    title="AnomalyOps-Copilot API",
    description="API RAG answers.",
    version="0.1.0",
)


@app.get("/health")
def health_check():
    return {"status": "ok"}