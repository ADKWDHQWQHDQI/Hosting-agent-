"""Compliance Compass — Streamlit Chat UI
Built on Microsoft Agent Framework + Azure AI Foundry
Run: streamlit run app_ui.py
"""

import asyncio
import os
import sys
from datetime import datetime

# Windows: aiohttp requires SelectorEventLoop (not the default ProactorEventLoop)
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
from dotenv import load_dotenv

# ── MUST be the very first Streamlit call ───────────────────────────────────────
st.set_page_config(
    page_title="Compliance Compass",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded",
)

load_dotenv()

# ── Streamlit Cloud: inject secrets as environment variables ────────────────────
# On Streamlit Community Cloud there is no .env file and no `az login`.
# Secrets defined in the Streamlit Cloud dashboard are available via st.secrets.
# We push them into os.environ so DefaultAzureCredential's EnvironmentCredential
# picks them up automatically — no code change needed between local and cloud.
_SECRET_KEYS = [
    "AZURE_AI_PROJECT_ENDPOINT",
    "AZURE_TENANT_ID",
    "AZURE_CLIENT_ID",
    "AZURE_CLIENT_SECRET",
]
for _k in _SECRET_KEYS:
    if _k in st.secrets and _k not in os.environ:
        os.environ[_k] = st.secrets[_k]

# ── aiohttp / azure-core transport patch (same root fix as run_agent.py) ────────
# azure-core creates its aiohttp ClientSession with auto_decompress=False,
# causing gzip/brotli-encoded Foundry responses to fail ContentDecodePolicy.
from azure.core.pipeline.transport._aiohttp import AioHttpTransport as _AioHttpTransport

_orig_open = _AioHttpTransport.open


async def _patched_open(self):
    if not self.session and self._session_owner:
        import aiohttp
        self.session = aiohttp.ClientSession(
            trust_env=self._use_env_settings,
            cookie_jar=aiohttp.DummyCookieJar(),
            auto_decompress=True,
        )
        self._has_been_opened = True
        await self.session.__aenter__()
    else:
        await _orig_open(self)


_AioHttpTransport.open = _patched_open

from agent_framework import Agent  # noqa: E402
from agent_framework.azure import AzureAIClient  # noqa: E402
from azure.ai.projects.aio import AIProjectClient  # noqa: E402
from azure.identity.aio import DefaultAzureCredential  # noqa: E402

# ── Constants ───────────────────────────────────────────────────────────────────
AGENT_NAME = "ComplianceAgent"
MODEL_DEPLOYMENT = "gpt-4o"

SAMPLE_QUERIES = [
    "Assess vendor risk for an AI company in Singapore processing payment data",
    "GDPR compliance requirements for transferring data to China",
    "Review contract clause: Customer data stored on servers in Frankfurt with backups in Mumbai",
    "What RBI regulations apply to third-party fintech vendors?",
]

# ── CSS ─────────────────────────────────────────────────────────────────────────
st.markdown(
    """
<style>
/* Page background */
[data-testid="stAppViewContainer"] > .main { background: #eef2f7; }
.main .block-container { padding-top: 1.5rem; padding-bottom: 2rem; max-width: 860px; }

/* Header banner */
.cc-header {
    background: linear-gradient(135deg, #0f2444 0%, #1e3a6e 60%, #2b5797 100%);
    border-radius: 14px;
    padding: 1.4rem 2rem;
    margin-bottom: 1.5rem;
    box-shadow: 0 4px 16px rgba(15,36,68,0.18);
}
.cc-header h1 { color: #ffffff; margin: 0; font-size: 1.8rem; font-weight: 700; letter-spacing: -0.01em; }
.cc-header p  { color: #93b8d9; margin: 0.35rem 0 0; font-size: 0.9rem; letter-spacing: 0.04em; text-transform: uppercase; }

/* Sidebar */
[data-testid="stSidebar"] > div:first-child { background: #0f2444; padding-top: 1rem; }
[data-testid="stSidebar"] .stMarkdown,
[data-testid="stSidebar"] .stMetric,
[data-testid="stSidebar"] label { color: #c8d8ea !important; }
[data-testid="stSidebar"] hr { border-color: #1e3a6e; }

/* Sidebar sample query buttons */
[data-testid="stSidebar"] .stButton > button {
    background: #1a3260;
    border: 1px solid #2e4f8a;
    color: #d0e4f4 !important;
    border-radius: 8px;
    text-align: left;
    white-space: normal;
    height: auto;
    font-size: 0.82rem;
    padding: 0.55rem 0.8rem;
    line-height: 1.4;
    transition: background 0.15s;
}
[data-testid="stSidebar"] .stButton > button:hover { background: #243f78; border-color: #4a72b5; }

/* Clear chat button — distinct colour */
[data-testid="stSidebar"] .stButton > button[kind="secondary"],
[data-testid="stSidebar"] .stButton + .stButton > button {
    background: #3b1a1a;
    border-color: #7a2c2c;
    color: #f5c0c0 !important;
}
[data-testid="stSidebar"] .stButton + .stButton > button:hover { background: #531f1f; }

/* Metric value colour */
[data-testid="stMetricValue"] { color: #93b8d9 !important; }

/* Chat message styling */
[data-testid="stChatMessage"] {
    border-radius: 12px;
    padding: 0.25rem 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# ── Session state ───────────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "query_count" not in st.session_state:
    st.session_state.query_count = 0
if "pending_query" not in st.session_state:
    st.session_state.pending_query = None


# ── Agent runner ────────────────────────────────────────────────────────────────
async def _run_async(user_input: str, placeholder) -> str:
    """Connect to the Foundry agent and stream the response into `placeholder`."""
    accumulated = ""
    try:
        endpoint = os.environ["AZURE_AI_PROJECT_ENDPOINT"]
    except KeyError:
        raise RuntimeError(
            "AZURE_AI_PROJECT_ENDPOINT is not set. "
            "Add it to your .env file and restart the app."
        )

    async with (
        DefaultAzureCredential() as credential,
        AIProjectClient(endpoint=endpoint, credential=credential) as project_client,
        Agent(
            client=AzureAIClient(
                project_client=project_client,
                agent_name=AGENT_NAME,
                model_deployment_name=MODEL_DEPLOYMENT,
                use_latest_version=True,
            ),
        ) as agent,
    ):
        tool_calls_seen: set = set()
        async for chunk in agent.run([user_input], stream=True):
            for call in (c for c in chunk.contents if c.type == "function_call"):
                if call.call_id not in tool_calls_seen:
                    tool_calls_seen.add(call.call_id)
                    accumulated += f"\n_🔧 Using tool: `{call.name}`_\n\n"
                    placeholder.markdown(accumulated + "▌")
            if chunk.text:
                accumulated += chunk.text
                placeholder.markdown(accumulated + "▌")

    placeholder.markdown(accumulated)
    return accumulated


def query_agent(user_input: str, placeholder) -> str:
    """Synchronous wrapper so Streamlit can call the async agent."""
    return asyncio.run(_run_async(user_input, placeholder))


# ── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🛡️ Compliance Compass")
    st.divider()

    st.markdown("#### About")
    st.markdown(
        "AI-powered regulatory risk assessment built on "
        "**Azure AI Foundry** and the **Microsoft Agent Framework**.\n\n"
        "Covers:\n"
        "- 🌍 Cross-border data transfer (GDPR, Schrems II, SCCs)\n"
        "- 🏦 RBI, PCI-DSS, CCPA, SOC 2\n"
        "- 🤝 Vendor & third-party risk\n"
        "- 📄 Contract clause analysis"
    )

    st.divider()
    st.markdown("#### 💡 Sample Queries")
    st.caption("Click to send instantly")
    for q in SAMPLE_QUERIES:
        if st.button(q, key=f"sq_{q[:32]}", use_container_width=True):
            st.session_state.pending_query = q
            st.rerun()

    st.divider()
    st.markdown("#### 📊 Session Info")
    st.metric("Queries this session", st.session_state.query_count)
    st.markdown("")
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.session_state.query_count = 0
        st.session_state.pending_query = None
        st.rerun()

# ── Header banner ───────────────────────────────────────────────────────────────
st.markdown(
    """
<div class="cc-header">
  <h1>🛡️ Compliance Compass</h1>
  <p>Regulatory Risk Assessment &amp; Policy Analysis</p>
</div>
""",
    unsafe_allow_html=True,
)

# ── Render chat history ──────────────────────────────────────────────────────────
for msg in st.session_state.messages:
    avatar = "🛡️" if msg["role"] == "assistant" else None
    with st.chat_message(msg["role"], avatar=avatar):
        st.markdown(msg["content"])
        st.caption(msg.get("timestamp", ""))

# ── Resolve input: sidebar click takes priority over typed input ─────────────────
pending = st.session_state.pending_query
st.session_state.pending_query = None

typed = st.chat_input("Ask a compliance or regulatory question…")
user_input = pending or typed

# ── Process and stream ───────────────────────────────────────────────────────────
if user_input:
    ts = datetime.now().strftime("%H:%M · %b %d")

    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input, "timestamp": ts})
    with st.chat_message("user"):
        st.markdown(user_input)
        st.caption(ts)

    # Stream agent response
    with st.chat_message("assistant", avatar="🛡️"):
        placeholder = st.empty()
        placeholder.markdown("_⏳ Analyzing compliance scenario…_")
        try:
            response = query_agent(user_input, placeholder)
            ts_resp = datetime.now().strftime("%H:%M · %b %d")
            st.caption(ts_resp)
            st.session_state.messages.append(
                {"role": "assistant", "content": response, "timestamp": ts_resp}
            )
            st.session_state.query_count += 1
        except Exception as exc:
            placeholder.error(
                f"⚠️ **Could not reach the agent.**\n\n"
                f"`{type(exc).__name__}: {exc}`\n\n"
                "**Checklist:**\n"
                "1. `.env` contains `AZURE_AI_PROJECT_ENDPOINT`\n"
                "2. You are authenticated — run `az login` in the terminal\n"
                "3. The `ComplianceAgent` exists in your Foundry project"
            )
