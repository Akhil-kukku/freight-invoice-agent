from fastapi import FastAPI
import uvicorn

app = FastAPI(title="AI Agent API")


@app.get("/")
def root():
    return {"message": "AI Agent API is running"}


if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
