# uvicorn api:app --reload
# uvicorn api:app

# TODO replace sub/pub with redis streams
# TODO port api service to go?

from fastapi import FastAPI
from pydantic import BaseModel
#from typing import Union
import uuid
import redis
from constants import QUEUE_CHANNEL_NAME, LOADER_CHANNEL_NAME
import json 

class Query(BaseModel):
    qry: str
    uuid: str

class Payload(BaseModel):
    file: str
    folder: str

app = FastAPI()
r = redis.StrictRedis('localhost', 6379, charset="utf-8", decode_responses=True)
@app.post("/")
async def create(qry: Query):
    qry.uuid = str(uuid.uuid4())
    r.publish(QUEUE_CHANNEL_NAME, json.dumps(qry.dict()))
    return qry

@app.patch("/")
async def load(payload: Payload):
    r.publish(LOADER_CHANNEL_NAME, json.dumps(payload.dict()))
    return {"status": "OK"}

@app.get("/{uuid}")
async def read(uuid):
    return {"uuid": uuid, "reply": r.hget("uuid",uuid)}

@app.get("/")
async def status():
    return {"status": "OK"}
