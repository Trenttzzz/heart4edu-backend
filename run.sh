#!/usr/bin/env bash
export PORT=8000
uvicorn app:app --host 0.0.0.0 --port $PORT --reload
