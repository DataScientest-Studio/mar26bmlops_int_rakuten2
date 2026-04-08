from fastapi import FastAPI

app = FastAPI(title="Rakuten Color Classification API")


@app.get("/")
def healthcheck():
    return {"status": "ok", "message": "API is running"}