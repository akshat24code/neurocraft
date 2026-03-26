# Deploy NeuroCraft Lab on Streamlit Cloud

## 1) Push code to GitHub
- Create a GitHub repository (or use an existing one).
- Push this project folder to your repo.
- Do not commit your real `.env` file.

## 2) Verify project files
- Entry file: `neurocraft_app.py`
- Dependencies: `requirements.txt`
- Optional apt packages file: `packages.txt`
- Example env file: `.env.example` should contain placeholders only, never real keys.

## 3) Deploy on Streamlit Cloud
1. Open [https://share.streamlit.io/](https://share.streamlit.io/)
2. Sign in with GitHub.
3. Click **New app**.
4. Select:
   - Repository: your repo
   - Branch: your deploy branch (usually `main`)
   - Main file path: `neurocraft_app.py`
5. Click **Deploy**.

## 4) Add API key securely
- Open your app in Streamlit Cloud.
- Go to **Settings -> Secrets**.
- Add this secret:

```toml
NVIDIA_API_KEY = "your_real_nvidia_key"
```

- Save and reboot the app.

## 5) Local vs Cloud key handling
- Local development: keep your real key in `.env`
- Streamlit Cloud: keep your real key in **Secrets**
- GitHub repo: commit `.env.example`, not `.env`

Example local `.env`:

```env
NVIDIA_API_KEY=your_real_nvidia_key
```

Example `.env.example`:

```env
# Copy this file to .env and fill your secrets
NVIDIA_API_KEY=your_nvidia_api_key_here
```

## 6) Security note
- If a real API key was ever committed to `.env.example`, Git history, screenshots, or chats, rotate that key immediately.
- After rotating, update your local `.env` and Streamlit Secrets with the new key.

## 7) Post-deploy smoke test
- Open `System Health` page in app.
- `.env File` may show as optional on deployment.
- Confirm `NVIDIA API Key` status is OK.
- Test `AI Playground -> Explore Data`.
- Test any LLM-backed text refinement flow.

## 8) Common cloud issues
- App fails to boot: check `requirements.txt` package compatibility.
- API key missing: ensure key is in Streamlit Secrets.
- `.env` warning on cloud: expected; use Secrets instead.
- Webcam issues in cloud: use WebRTC mode and HTTPS app URL.
