from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from trainer import run_local_training
from db import init_db
import logging

app = FastAPI(title="Rapha Enterprise Node (TEE Mock)")
logging.basicConfig(level=logging.INFO)

class TrainingPayload(BaseModel):
    dataset_id: str
    weights: str
    job_id: str

@app.on_event("startup")
def startup_event():
    init_db()
    logging.info("SQLite Mock EHR DB Initialized.")

@app.get("/health")
def health_check():
    return {"status": "Rapha Node Active", "version": "1.0.0"}

@app.post("/train")
def receive_payload_and_train(payload: TrainingPayload):
    logging.info(f"Received payload for Job {payload.job_id}, Dataset {payload.dataset_id}")
    try:
        updated_weights = run_local_training(payload.weights)
        # Mock ZK proof generation
        zk_proof = f"zk_snark_proof_{payload.job_id}_valid"
        return {"updated_weights": updated_weights, "zk_proof": zk_proof}
    except Exception as e:
        logging.error(f"Training failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))
