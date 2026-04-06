"""
server/app.py — FastAPI server using openenv_core.

create_app() wires up /reset, /step, /state, /health, /ws, /web.
GET / added so HuggingFace Space health probes return 200.
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_app

from models import DataCleaningAction, DataCleaningObservation
from server.dc_environment import DataCleaningEnvironment

env = DataCleaningEnvironment

app = create_app(env, DataCleaningAction, DataCleaningObservation)


@app.get("/")
def root():
    return {
        "name":    "Data Cleaning OpenEnv",
        "version": "0.1.0",
        "status":  "running",
        "tasks":   ["ecommerce_easy", "patient_records_medium", "financial_audit_hard"],
        "endpoints": {
            "health":    "/health",
            "reset":     "POST /reset",
            "step":      "POST /step",
            "state":     "GET  /state",
            "websocket": "/ws",
            "docs":      "/docs",
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=7860, reload=True)