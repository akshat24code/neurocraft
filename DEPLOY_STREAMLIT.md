# Deploy NeuroCraft Lab on Streamlit Cloud

## 1) Push code to GitHub
- Create a GitHub repository (or use an existing one).
- Push this project folder to your repo.

## 2) Verify project files
- Entry file: `neurocraft_app.py`
- Dependencies: `requirements.txt`
- Optional apt packages file: `packages.txt`

## 3) Deploy on Streamlit Cloud
1. Open [https://share.streamlit.io/](https://share.streamlit.io/)
2. Sign in with GitHub.
3. Click **New app**.
4. Select:
   - Repository: your repo
   - Branch: your deploy branch (usually `main`)
   - Main file path: `neurocraft_app.py`
5. Click **Deploy**.

## 4) Add API key securely (important)
- Open your app in Streamlit Cloud.
- Go to **Settings -> Secrets**.
- Add this secret:

```toml
NVIDIA_API_KEY = "your_real_nvidia_key"
```

- Save and reboot the app.

## 5) Local vs Cloud key handling
- Local development: use `.env` with `NVIDIA_API_KEY=...`
- Streamlit Cloud: use **Secrets** (not `.env`)

## 6) Post-deploy smoke test
- Open `System Health` page in app.
- Confirm key status and asset checks are OK.
- Test `AI Playground -> Explore Data`.
- Upload `IRIS.csv` and verify AI summary appears.

## 7) Common cloud issues
- App fails to boot: check `requirements.txt` package compatibility.
- API key missing: ensure key is in Streamlit Secrets.
- Webcam issues in cloud: use WebRTC mode and HTTPS app URL.
