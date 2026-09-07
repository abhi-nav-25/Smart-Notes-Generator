from fastapi import FastAPI

app = FastAPI(title="Smart Notes Generator - ML Service")


@app.get("/health")
def health_check():
    return {"status": "ok", "message": "ML service is running"}
