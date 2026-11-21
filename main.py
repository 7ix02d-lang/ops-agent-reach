from fastapi import FastAPI
from pydantic import BaseModel
import subprocess
import os

app = FastAPI(title="OPS Agent Reach API")

class SearchRequest(BaseModel):
    query: str
    platform: str = "youtube"

@app.post("/search")
def search(data: SearchRequest):
    try:
        cmd = f'agent-reach {data.platform} search "{data.query}"'
        result = subprocess.getoutput(cmd)
        return {"result": result}
    except Exception as e:
        return {"error": str(e)}

@app.get("/")
def home():
    return {"status": "Agent Reach is working"}

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run(app, host="0.0.0.0", port=port)
