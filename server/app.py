"""
server/app.py — FastAPI server using the real OpenEnv framework.

Uses create_fastapi_app() from openenv.core.env_server, which automatically
wires up /reset, /step, /state, /health, /ws (WebSocket), and /web endpoints.
"""

import sys
import os

# Ensure the project root is importable
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from openenv.core.env_server import create_fastapi_app

from models import DataCleaningAction, DataCleaningObservation
from server.dc_environment import DataCleaningEnvironment

env = DataCleaningEnvironment()

app = create_fastapi_app(env, DataCleaningAction, DataCleaningObservation)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("server.app:app", host="0.0.0.0", port=8000, reload=True)