"""Start de server: `python run.py`. Poort via BACKTEST_PORT (standaard 9000, want 8000 is de portfolio server)."""
import os

import uvicorn

if __name__ == "__main__":
    uvicorn.run("app.main:app", host=os.getenv("BACKTEST_HOST", "0.0.0.0"), port=int(os.getenv("BACKTEST_PORT", "9000")))
