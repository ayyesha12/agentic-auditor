# Deployment

## Public API (ngrok)

The API is exposed publicly via ngrok tunneling from a local machine.

Start sequence:
1. Ollama runs automatically on Windows startup (port 11434)
2. Start API: `uvicorn api.main:app --port 8000`
3. Start dashboard: `streamlit run dashboard/app.py`
4. Expose publicly: `.\ngrok.exe http 8000`

Public URL: generated fresh each time ngrok starts (free tier)

## Endpoints

- GET  /health  — confirms API and graph are loaded
- POST /ask     — runs a question through the 3-agent pipeline
- POST /run-eval — triggers a full eval batch

## Local URLs

- API:       http://localhost:8000
- Dashboard: http://localhost:8501
- API docs:  http://localhost:8000/docs

## Limitations

- ngrok free tier generates a new URL each session
- Upgrade to ngrok paid tier for a fixed URL
- AWS EC2 deployment pending account activation