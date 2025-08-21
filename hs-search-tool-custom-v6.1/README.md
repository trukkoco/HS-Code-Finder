# HS Code Finder — v6.1 (fixed keyword regex)

## Run
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt

# API
python -m uvicorn app.api:app --reload

# UI (new PowerShell)
.\.venv\Scripts\Activate.ps1
streamlit run ui/app.py
```
- API: http://localhost:8000/health
- UI:  http://localhost:8501
