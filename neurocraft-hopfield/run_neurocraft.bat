@echo off
echo Starting NeuroCraft - Hopfield Memory Lab...

start cmd /k "cd backend && python -m venv venv && venv\Scripts\activate && pip install -r requirements.txt && python app.py"
start cmd /k "cd frontend && npm install && npm run dev"

echo Servers are starting in separate windows.
echo Backend: http://localhost:5000
echo Frontend: http://localhost:5173
pause
