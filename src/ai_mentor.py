import os
import requests
import streamlit as st

def _get_secret_or_env(name: str) -> str:
    value = os.getenv(name, '')
    if value:
        return value
    try:
        return st.secrets.get(name, '')
    except Exception:
        return ''

def call_nvidia_nim(prompt: str, context: str, api_key: str) -> str:
    url = "https://integrate.api.nvidia.com/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    mode = st.session_state.get('learning_mode', 'Explorer')
    system_prompt = (
        f"You are an AI Mentor for a neural network learning platform. The user is currently studying: {context}. "
        f"The current learning mode is set to: {mode}. "
        f"{'Explain like I am 5, focus on intuition and metaphors.' if mode == 'Beginner' else ''}"
        f"{'Provide interactive and exploratory advice, suggesting experiment parameters.' if mode == 'Explorer' else ''}"
        f"{'Focus on the underlying mathematics, linear algebra, and logic. Use rigorous terminology.' if mode == 'Research' else ''}"
        "Keep responses concise, educational, and encouraging."
    )
    payload = {
        "model": "meta/llama-3.1-70b-instruct",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 512,
        "temperature": 0.6,
    }
    try:
        resp = requests.post(url, headers=headers, json=payload, timeout=30)
        resp.raise_for_status()
        return resp.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        return f"Error communicating with AI Mentor API: {e}"

def render_ai_mentor(context_text: str):
    with st.sidebar.expander("🤖 Ask AI Mentor", expanded=False):
        st.write("I am your AI Mentor! Ask me anything about the current module.")
        
        if "mentor_messages" not in st.session_state:
            st.session_state.mentor_messages = []
            
        for msg in st.session_state.mentor_messages[-3:]: # Show last 3 messages
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                
        def submit_chat():
            user_msg = st.session_state.mentor_input
            if user_msg:
                st.session_state.mentor_messages.append({"role": "user", "content": user_msg})
                
                api_key = _get_secret_or_env("NVIDIA_API_KEY")
                if api_key:
                    response = call_nvidia_nim(user_msg, context_text, api_key)
                else:
                    response = ("I'm currently running in offline simulation mode. "
                                "To activate full AI capabilities, please add the NVIDIA_API_KEY to your environment variables or Streamlit secrets.")
                
                st.session_state.mentor_messages.append({"role": "assistant", "content": response})
                st.session_state.mentor_input = ""
                
        st.text_input("Ask a question...", key="mentor_input", on_change=submit_chat)
