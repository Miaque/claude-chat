import os

import uvicorn
from fastapi import FastAPI

os.environ["TZ"] = "Asia/Shanghai"

app = FastAPI()


@app.get("/")
async def root():
    return {"message": "Hello World"}


@app.get("/hello/{name}")
async def say_hello(name: str):
    return {"message": f"Hello {name}"}


if __name__ == "__main__":
    uvicorn.run("api:app", host="0.0.0.0", port=8080)
