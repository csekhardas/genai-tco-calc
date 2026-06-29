import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from models.pricing_data import (
    GPU_PRICING, LLM_API_PRICING, LABOR_RATES, VECTOR_DB_PRICING,
    OPEN_SOURCE_MODELS, FINE_TUNING_GPU_HOURS_PER_BILLION_TOKENS,
    LEGACY_SYSTEM_BENCHMARKS,
)
from models.cost_model import (
    compute_api_model_cost, compute_gpu_monthly_cost,
    compute_infrastructure_cost, compute_data_pipeline_cost,
    compute_labor_cost, compute_maintenance_cost,
    compute_total_tco, compute_scenario_comparison,
    compute_fine_tuning_cost, project_costs_over_time,
)
from models.roi_model import (
    compute_productivity_gains, compute_cost_avoidance,
    compute_revenue_impact, compute_full_roi,
)

st.set_page_config(
    page_title="GenAI TCO Calculator",
    page_icon="☁️",
    layout="wide",
    initial_sidebar_state="expanded",
)

GC = dict(
    blue       = "#1A73E8",
    blue_dark  = "#174EA6",
    blue_light = "#E8F0FE",
    green      = "#34A853",
    green_light= "#E6F4EA",
    yellow     = "#FBBC04",
    yellow_light="#FEF7E0",
    red        = "#EA4335",
    red_light  = "#FCE8E6",
    purple     = "#9334E6",
    cyan       = "#00ACC1",
    text       = "#202124",
    muted      = "#5F6368",
    border     = "#DADCE0",
    bg         = "#F8FAFD",
    card       = "#FFFFFF",
)
COLORS = [GC["blue"], GC["green"], GC["red"], GC["yellow"], GC["purple"], GC["cyan"]]


def chart_layout(**kw):
    base = dict(
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(248,250,253,0.5)",
        font=dict(family="Roboto, Arial, sans-serif", color=GC["muted"], size=12),
        xaxis=dict(gridcolor="#F1F3F4", zerolinecolor=GC["border"], linecolor=GC["border"]),
        yaxis=dict(gridcolor="#F1F3F4", zerolinecolor=GC["border"], linecolor=GC["border"]),
        legend=dict(bgcolor="rgba(255,255,255,0.95)", bordercolor=GC["border"], borderwidth=1,
                    font=dict(color=GC["text"])),
        margin=dict(t=24, b=24, l=8, r=8),
        hoverlabel=dict(bgcolor=GC["card"], bordercolor=GC["border"],
                        font_color=GC["text"], font_family="Roboto, Arial, sans-serif"),
    )
    base.update(kw)
    return base


def fmt(n):
    if abs(n) >= 1_000_000:
        return f"${n/1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"${n/1_000:.1f}K"
    return f"${n:.0f}"


# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Google+Sans:wght@400;500;700&family=Roboto:wght@300;400;500;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Roboto', Arial, sans-serif;
    color: #202124;
    background: #F8FAFD;
}

.block-container { max-width: 1380px; padding-top: 1.2rem; padding-bottom: 3rem; }

/* ── Top bar ── */
.gc-topbar {
    height: 64px; background: #FFFFFF; border: 1px solid #DADCE0;
    border-radius: 16px; padding: 0 28px;
    display: flex; align-items: center; justify-content: space-between;
    margin-bottom: 20px; box-shadow: 0 1px 3px rgba(60,64,67,0.10);
}
.gc-brand { display:flex; align-items:center; gap:10px;
    font-family:'Google Sans','Roboto',Arial,sans-serif;
    font-size:19px; font-weight:700; color:#202124; }
.gc-cloud-icon { font-size:26px; }
.gc-nav { display:flex; gap:8px; }
.gc-nav-item {
    padding: 8px 18px; border-radius: 8px; font-size:14px; font-weight:500;
    color:#5F6368; cursor:default;
}

/* ── Section divider label ── */
.gc-tab-label {
    font-family:'Google Sans','Roboto',Arial,sans-serif;
    font-size:11px; font-weight:700; color:#5F6368; letter-spacing:1.5px;
    text-transform:uppercase; margin: 4px 0 8px;
}

/* ── Hero ── */
.gc-hero {
    background: #FFFFFF; border: 1px solid #DADCE0; border-radius: 24px;
    padding: 44px 48px; margin-bottom: 24px;
    box-shadow: 0 1px 3px rgba(60,64,67,0.10); position: relative; overflow: hidden;
}
.gc-hero::after {
    content: ""; position: absolute; right: -80px; top: -80px;
    width: 280px; height: 280px;
    background: radial-gradient(circle, rgba(26,115,232,0.10), transparent 65%);
    pointer-events: none;
}
.gc-eyebrow {
    display: inline-flex; align-items: center; gap: 7px;
    background: #E8F0FE; color: #174EA6;
    padding: 7px 14px; border-radius: 999px;
    font-size: 13px; font-weight: 700; margin-bottom: 20px;
}
.gc-hero h1 {
    font-family: 'Google Sans', 'Roboto', Arial, sans-serif;
    font-size: 38px; line-height: 1.12; letter-spacing: -0.01em;
    color: #202124; margin: 0 0 14px; max-width: 760px;
}
.gc-hero p { color: #5F6368; font-size: 17px; line-height: 1.65; max-width: 700px; margin-bottom: 26px; }
.gc-cta-row { display: flex; gap: 12px; flex-wrap: wrap; }
.gc-btn-primary {
    display: inline-block; background: #1A73E8; color: #FFFFFF !important;
    padding: 11px 22px; border-radius: 8px; font-weight: 700; font-size: 14px;
    text-decoration: none; cursor: pointer;
}
.gc-btn-secondary {
    display: inline-block; background: #FFFFFF; color: #1A73E8 !important;
    padding: 10px 22px; border-radius: 8px; font-weight: 700; font-size: 14px;
    border: 1px solid #DADCE0; text-decoration: none; cursor: pointer;
}

/* ── KPI Cards ── */
.gc-card {
    background: #FFFFFF; border: 1px solid #DADCE0; border-radius: 16px;
    padding: 22px 22px 18px; box-shadow: 0 1px 2px rgba(60,64,67,0.08);
    transition: box-shadow 0.2s; height: 100%; min-height: 130px;
}
.gc-card:hover { box-shadow: 0 4px 14px rgba(60,64,67,0.16); }
.gc-blue   { border-top: 4px solid #1A73E8; }
.gc-green  { border-top: 4px solid #34A853; }
.gc-yellow { border-top: 4px solid #FBBC04; }
.gc-red    { border-top: 4px solid #EA4335; }
.gc-purple { border-top: 4px solid #9334E6; }
.gc-cyan   { border-top: 4px solid #00ACC1; }
.gc-kpi-label { color: #5F6368; font-size: 11px; font-weight: 700;
    letter-spacing: 1.2px; text-transform: uppercase; margin-bottom: 10px; }
.gc-kpi-value { font-family:'Google Sans','Roboto',Arial,sans-serif;
    color: #202124; font-size: 32px; font-weight: 700; line-height: 1; margin-bottom: 8px; }
.gc-kpi-sub { color: #5F6368; font-size: 13px; line-height: 1.4; }

/* ── Info cards (for TCO pillars, challenges, etc.) ── */
.gc-info-card {
    background: #FFFFFF; border: 1px solid #DADCE0; border-radius: 14px;
    padding: 20px 20px 16px; height: 100%;
    box-shadow: 0 1px 2px rgba(60,64,67,0.06);
}
.gc-info-card:hover { box-shadow: 0 3px 10px rgba(60,64,67,0.14); }
.gc-info-icon  { font-size: 26px; margin-bottom: 10px; }
.gc-info-title { font-family:'Google Sans','Roboto',Arial,sans-serif;
    font-size: 15px; font-weight: 700; color: #202124; margin-bottom: 6px; }
.gc-info-body  { font-size: 13px; color: #5F6368; line-height: 1.6; }

/* ── Challenge cards ── */
.gc-challenge {
    background: #FFFFFF; border: 1px solid #DADCE0; border-left: 4px solid #EA4335;
    border-radius: 12px; padding: 14px 18px; margin: 6px 0;
    box-shadow: 0 1px 2px rgba(60,64,67,0.06);
}
.gc-challenge-title { font-weight: 700; color: #202124; font-size: 14px; margin-bottom: 4px; }
.gc-challenge-body  { color: #5F6368; font-size: 13px; line-height: 1.55; }

/* ── Scenario info cards ── */
.gc-scenario-card {
    background: #FFFFFF; border: 1px solid #DADCE0; border-radius: 16px;
    padding: 24px; height: 100%;
    box-shadow: 0 1px 2px rgba(60,64,67,0.08);
}
.gc-scenario-card:hover { box-shadow: 0 4px 14px rgba(60,64,67,0.14); }
.gc-scenario-badge {
    display: inline-block; padding: 4px 12px; border-radius: 999px;
    font-size: 11px; font-weight: 700; margin-bottom: 12px;
}
.gc-scenario-title { font-family:'Google Sans','Roboto',Arial,sans-serif;
    font-size: 17px; font-weight: 700; color: #202124; margin-bottom: 8px; }
.gc-scenario-desc { font-size: 13px; color: #5F6368; line-height: 1.6; margin-bottom: 14px; }
.gc-tag-row { display:flex; gap:6px; flex-wrap:wrap; margin-top:10px; }
.gc-tag {
    background: #F1F3F4; color: #5F6368; padding: 3px 10px;
    border-radius: 999px; font-size: 11px; font-weight: 500;
}

/* ── Optimization technique cards ── */
.gc-opt-card {
    background: #FFFFFF; border: 1px solid #DADCE0; border-radius: 14px;
    padding: 20px; height: 100%;
    box-shadow: 0 1px 2px rgba(60,64,67,0.06);
}
.gc-opt-card:hover { box-shadow: 0 3px 10px rgba(60,64,67,0.14); }
.gc-opt-title { font-family:'Google Sans','Roboto',Arial,sans-serif;
    font-size: 15px; font-weight: 700; color: #202124; margin-bottom: 6px; }
.gc-opt-body  { font-size: 13px; color: #5F6368; line-height: 1.6; }
.gc-opt-list  { padding-left: 16px; margin: 8px 0 0; }
.gc-opt-list li { font-size: 12px; color: #5F6368; margin-bottom: 3px; }

/* ── Callout box ── */
.gc-callout {
    background: #E8F0FE; border: 1px solid #C5D9F7; border-radius: 14px;
    padding: 20px 24px; margin: 16px 0;
}
.gc-callout-title { font-family:'Google Sans','Roboto',Arial,sans-serif;
    font-size: 16px; font-weight: 700; color: #174EA6; margin-bottom: 6px; }
.gc-callout-body { font-size: 14px; color: #174EA6; line-height: 1.65; }

/* ── Tabs ── */
div[data-baseweb="tab-list"] {
    background: #FFFFFF !important; border: 1px solid #DADCE0 !important;
    border-radius: 14px !important; padding: 6px !important; gap: 4px !important;
    box-shadow: 0 1px 2px rgba(60,64,67,0.08) !important; margin-bottom: 24px;
}
button[data-baseweb="tab"] {
    border-radius: 10px !important; padding: 10px 18px !important;
    color: #5F6368 !important; font-weight: 500 !important;
    background: transparent !important; border: none !important;
    font-size: 14px !important; transition: background 0.15s !important;
}
button[data-baseweb="tab"]:hover { background: #F8FAFD !important; color: #202124 !important; }
button[data-baseweb="tab"][aria-selected="true"] {
    background: #E8F0FE !important; color: #1A73E8 !important; font-weight: 700 !important;
}
button[data-baseweb="tab"] p { font-size: 14px !important; }
div[data-baseweb="tab-panel"] { padding-top: 4px !important; }

/* ── Section titles ── */
.gc-section-title {
    font-family: 'Google Sans','Roboto',Arial,sans-serif;
    color: #202124; font-size: 22px; font-weight: 700;
    margin: 4px 0 6px; line-height: 1.3;
}
.gc-section-sub { color: #5F6368; font-size: 14px; margin-bottom: 20px; line-height: 1.6; }
.gc-sub-hdr {
    font-size: 13px; font-weight: 700; color: #5F6368; letter-spacing: 1px;
    text-transform: uppercase; padding: 18px 0 10px;
    border-bottom: 1px solid #F1F3F4; margin-bottom: 14px; display: flex;
    align-items: center; gap: 8px;
}
.gc-sub-hdr::before {
    content: ''; width: 3px; height: 14px;
    background: linear-gradient(180deg, #1A73E8, #34A853); border-radius: 2px;
}

/* ── Recommendations ── */
.gc-rec {
    background: #FFFFFF; border: 1px solid #DADCE0; border-left: 4px solid #1A73E8;
    border-radius: 12px; padding: 14px 18px; margin: 8px 0;
    box-shadow: 0 1px 2px rgba(60,64,67,0.06); transition: box-shadow 0.2s;
}
.gc-rec:hover { box-shadow: 0 2px 8px rgba(60,64,67,0.14); }
.gc-rec-title { font-weight: 700; color: #202124; font-size: 14px; margin-bottom: 4px; }
.gc-rec-body  { color: #5F6368; font-size: 13px; line-height: 1.6; }

/* ── Report section cards ── */
.gc-report-card {
    background: #FFFFFF; border: 1px solid #DADCE0; border-radius: 14px;
    padding: 20px; height: 100%;
    box-shadow: 0 1px 2px rgba(60,64,67,0.06);
}
.gc-report-card:hover { box-shadow: 0 3px 10px rgba(60,64,67,0.14); }
.gc-report-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 28px; height: 28px; border-radius: 50%;
    background: #E8F0FE; color: #174EA6; font-weight: 700; font-size: 13px;
    margin-bottom: 10px;
}
.gc-report-title { font-family:'Google Sans','Roboto',Arial,sans-serif;
    font-size: 15px; font-weight: 700; color: #202124; margin-bottom: 6px; }
.gc-report-body  { font-size: 13px; color: #5F6368; line-height: 1.6; }

/* ── Streamlit overrides ── */
[data-testid="stSidebar"] {
    background: #FFFFFF !important;
    border-right: 1px solid #DADCE0 !important;
}
[data-testid="stSidebar"] .stMarkdown h3 {
    color: #1A73E8 !important; font-size: 11px !important; font-weight: 700 !important;
    letter-spacing: 1.5px !important; text-transform: uppercase !important;
}
[data-testid="stSidebar"] hr { border-color: #DADCE0 !important; }

.stMetric {
    background: #FFFFFF !important; border: 1px solid #DADCE0 !important;
    border-radius: 12px !important; padding: 16px !important;
    box-shadow: 0 1px 2px rgba(60,64,67,0.06) !important;
}
[data-testid="stMetricValue"]  { color: #202124 !important; font-weight: 700 !important; }
[data-testid="stMetricLabel"]  { color: #5F6368 !important; font-size: 12px !important;
    font-weight: 700 !important; text-transform: uppercase !important; letter-spacing: 0.8px !important; }
[data-testid="stMetricDelta"]  { color: #34A853 !important; }

.stTextInput input, .stNumberInput input {
    border-radius: 8px !important; border: 1px solid #DADCE0 !important;
    color: #202124 !important; background: #FFFFFF !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color: #1A73E8 !important; box-shadow: 0 0 0 3px rgba(26,115,232,0.12) !important;
}
div[data-baseweb="select"] > div { border-radius: 8px !important; border: 1px solid #DADCE0 !important; }

.stButton > button {
    background: #1A73E8 !important; color: #FFFFFF !important;
    border: none !important; border-radius: 8px !important;
    font-weight: 700 !important; padding: 0.6rem 1.2rem !important;
    font-size: 14px !important; transition: background 0.15s !important;
}
.stButton > button:hover { background: #174EA6 !important; }

.stDownloadButton > button {
    background: #FFFFFF !important; color: #1A73E8 !important;
    border: 1px solid #DADCE0 !important; border-radius: 8px !important;
    font-weight: 700 !important; padding: 0.6rem 1.2rem !important;
}
.stDownloadButton > button:hover { background: #E8F0FE !important; }

.streamlit-expanderHeader {
    background: #FFFFFF !important; border: 1px solid #DADCE0 !important;
    border-radius: 10px !important; color: #202124 !important; font-weight: 500 !important;
}
.streamlit-expanderContent {
    background: #FAFAFA !important; border: 1px solid #DADCE0 !important;
    border-top: none !important; border-radius: 0 0 10px 10px !important;
}

div[data-baseweb="notification"] {
    background: #E8F0FE !important; border: 1px solid #C5D9F7 !important;
    border-radius: 10px !important;
}
div[data-baseweb="notification"] * { color: #174EA6 !important; }

[data-testid="stDataFrame"] {
    border: 1px solid #DADCE0 !important; border-radius: 12px !important;
    overflow: hidden !important;
}
hr { border-color: #F1F3F4 !important; }
[data-testid="stSlider"] > div > div > div > div { background: #1A73E8 !important; }
header[data-testid="stHeader"] { background: transparent !important; }
::-webkit-scrollbar { width: 6px; height: 6px; }
::-webkit-scrollbar-track { background: #F8FAFD; }
::-webkit-scrollbar-thumb { background: #DADCE0; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #BDC1C6; }
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def gc_card(label, value, sub, color="gc-blue"):
    st.markdown(f"""
    <div class="gc-card {color}">
        <div class="gc-kpi-label">{label}</div>
        <div class="gc-kpi-value">{value}</div>
        <div class="gc-kpi-sub">{sub}</div>
    </div>""", unsafe_allow_html=True)

def section(title, subtitle=""):
    st.markdown(f'<div class="gc-section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="gc-section-sub">{subtitle}</div>', unsafe_allow_html=True)

def sub_hdr(label):
    st.markdown(f'<div class="gc-sub-hdr">{label}</div>', unsafe_allow_html=True)

def rec(icon, title, body, color="#1A73E8"):
    st.markdown(f"""
    <div class="gc-rec" style="border-left-color:{color}">
        <div class="gc-rec-title">{icon} {title}</div>
        <div class="gc-rec-body">{body}</div>
    </div>""", unsafe_allow_html=True)

def info_card(icon, title, body):
    return f"""
    <div class="gc-info-card">
        <div class="gc-info-icon">{icon}</div>
        <div class="gc-info-title">{title}</div>
        <div class="gc-info-body">{body}</div>
    </div>"""

def challenge_card(icon, title, body):
    return f"""
    <div class="gc-challenge">
        <div class="gc-challenge-title">{icon} {title}</div>
        <div class="gc-challenge-body">{body}</div>
    </div>"""

def opt_card(icon, title, body, bullets):
    items = "".join(f"<li>{b}</li>" for b in bullets)
    return f"""
    <div class="gc-opt-card">
        <div style="font-size:22px;margin-bottom:8px">{icon}</div>
        <div class="gc-opt-title">{title}</div>
        <div class="gc-opt-body">{body}</div>
        <ul class="gc-opt-list">{items}</ul>
    </div>"""

def report_card(num, title, body, includes):
    items = "".join(f"<li>{i}</li>" for i in includes)
    return f"""
    <div class="gc-report-card">
        <div class="gc-report-num">{num}</div>
        <div class="gc-report-title">{title}</div>
        <div class="gc-report-body">{body}</div>
        <ul class="gc-opt-list" style="margin-top:10px">{items}</ul>
    </div>"""


# ─── Sidebar ─────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:10px;padding:6px 0 14px">
        <span style="font-size:30px">☁️</span>
        <div>
            <div style="font-family:'Google Sans',Roboto,Arial,sans-serif;font-size:16px;font-weight:700;color:#202124;">GenAI TCO</div>
            <div style="font-size:11px;color:#5F6368;letter-spacing:0.5px;">Enterprise Calculator</div>
        </div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("### Company Profile")
    company_name  = st.text_input("Company Name", "Acme Corp")
    industry      = st.selectbox("Industry", ["Financial Services","Healthcare","Retail / E-commerce",
                                               "Technology","Manufacturing","Telecom","Insurance","Other"])
    num_employees = st.number_input("Total Employees", 100, 500_000, 5000, step=500)

    st.markdown("### Use Case")
    use_case = st.selectbox("Primary Use Case", [
        "Customer Support Automation","Knowledge Base / Internal Q&A",
        "Document Processing & Summarization","Code Generation / Developer Tooling",
        "Sales & Marketing Personalization","Fraud Detection & Risk Analysis"])

    st.markdown("### Scale")
    monthly_queries   = st.number_input("Monthly AI Queries", 1000, 50_000_000, 500_000, step=10_000)
    avg_input_tokens  = st.slider("Avg Input Tokens / Query",  100, 8000, 800, step=100)
    avg_output_tokens = st.slider("Avg Output Tokens / Query",  50, 4000, 300, step=50)

    st.markdown("### Deployment")
    cloud_provider  = st.selectbox("Cloud Provider", ["GCP","AWS","Azure"])
    deployment_type = st.selectbox("Deployment Type", [
        "Cloud API (Managed)","Cloud VM (Self-managed)","Self-hosted (On-Prem)"])
    model_approach  = st.selectbox("Model Approach", [
        "API / Third-party LLM","Fine-tuned Open Source","Self-hosted OSS"])

    st.divider()
    st.markdown("""
    <div style="display:flex;gap:6px;flex-wrap:wrap">
        <span style="background:#E8F0FE;color:#174EA6;padding:3px 10px;border-radius:999px;font-size:11px;font-weight:700;">v1.0</span>
        <span style="background:#E6F4EA;color:#137333;padding:3px 10px;border-radius:999px;font-size:11px;font-weight:700;">Multi-Cloud</span>
        <span style="background:#FEF7E0;color:#B06000;padding:3px 10px;border-radius:999px;font-size:11px;font-weight:700;">NVIDIA GPU</span>
    </div>""", unsafe_allow_html=True)


# ─── Top bar ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="gc-topbar">
    <div class="gc-brand">
        <span class="gc-cloud-icon">☁️</span>
        <span>GenAI TCO Calculator</span>
    </div>
    <div class="gc-nav">
        <span class="gc-nav-item">Overview</span>
        <span class="gc-nav-item">Scenarios</span>
        <span class="gc-nav-item">Optimization</span>
        <span class="gc-nav-item">Reports</span>
    </div>
</div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# ROW 1 — 4 CONTENT TABS  (Overview · Scenarios · Optimization · Reports)
# ════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="gc-tab-label">Knowledge Base</div>', unsafe_allow_html=True)
tab_ov, tab_sc_info, tab_opt_info, tab_rep_info = st.tabs([
    "🏠 Overview",
    "🔄 Scenarios",
    "⚡ Optimization",
    "📋 Reports",
])

st.markdown('<div class="gc-tab-label" style="margin-top:20px">TCO Calculator</div>', unsafe_allow_html=True)
# ════════════════════════════════════════════════════════════════════════════
# ROW 2 — 6 CALCULATOR TABS  (original set, unchanged)
# ════════════════════════════════════════════════════════════════════════════
tab_dashboard, tab_cost, tab_scenario, tab_gpu, tab_roi, tab_export = st.tabs([
    "📊 Executive Dashboard",
    "💰 Cost Components",
    "🔄 Scenario Comparison",
    "🖥️ GPU Optimization",
    "📈 ROI & Payback",
    "📋 Docs & Export",
])


# ────────────────────────────────────────────────────────────────────────────
# CALCULATOR TAB 2 — Cost Components  (compute FIRST — feeds all other tabs)
# ────────────────────────────────────────────────────────────────────────────
with tab_cost:
    section("Cost Components",
            "Break down GenAI cost by infrastructure, model tokens, data pipeline, labor, and operations.")

    sub_hdr("Infrastructure")
    c1, c2, c3 = st.columns(3)
    with c1:
        gpu_type  = st.selectbox("GPU Type", list(GPU_PRICING.keys()), key="gpu_type")
        gpu_count = st.number_input("GPU Count", 1, 256, 4, key="gpu_count")
    with c2:
        utilization_pct = st.slider("GPU Utilization %", 10, 100, 60, key="util")
        pricing_tier    = st.selectbox("Pricing Tier", ["on_demand","reserved_1yr","reserved_3yr"], key="tier",
            format_func=lambda x:{"on_demand":"On-Demand","reserved_1yr":"Reserved 1yr","reserved_3yr":"Reserved 3yr"}[x])
    with c3:
        storage_gb            = st.number_input("Storage (GB)", 100, 500_000, 5000, step=100, key="storage_gb")
        networking_egress_gb  = st.number_input("Monthly Egress (GB)", 10, 100_000, 500, key="egress_gb")

    infra = compute_infrastructure_cost(
        deployment_type, gpu_type, gpu_count, utilization_pct,
        pricing_tier, storage_gb, cloud_provider, networking_egress_gb)

    sub_hdr("Model Cost")
    ca, cb = st.columns(2)
    with ca:
        if model_approach == "API / Third-party LLM":
            selected_model = st.selectbox("LLM Model", list(LLM_API_PRICING.keys()), key="api_model")
            model_api_cost = compute_api_model_cost(selected_model, monthly_queries, avg_input_tokens, avg_output_tokens)
            st.info(f"Token cost: **{fmt(model_api_cost)}/month** at {monthly_queries:,} queries")
        elif model_approach == "Fine-tuned Open Source":
            ft_model_size    = st.selectbox("Base Model Size", list(FINE_TUNING_GPU_HOURS_PER_BILLION_TOKENS.keys()), key="ft_size")
            training_tokens_b = st.slider("Training Data (Billion Tokens)", 0.1, 10.0, 1.0, step=0.1, key="train_tok")
            ft_one_time      = compute_fine_tuning_cost(ft_model_size, training_tokens_b, gpu_type, pricing_tier)
            model_api_cost   = compute_gpu_monthly_cost(gpu_type, max(1, gpu_count//2), utilization_pct, pricing_tier)
            st.info(f"One-time fine-tuning: **{fmt(ft_one_time)}** · Inference: **{fmt(model_api_cost)}/month**")
        else:
            oss_model      = st.selectbox("OSS Model", list(OPEN_SOURCE_MODELS.keys()), key="oss_model")
            model_api_cost = compute_gpu_monthly_cost(gpu_type, gpu_count, utilization_pct, pricing_tier)
            m = OPEN_SOURCE_MODELS[oss_model]
            st.info(f"~{m['tokens_per_sec_a100']} tokens/sec on A100 · {m['gpu_memory_gb']} GB VRAM required")
    with cb:
        if model_approach == "API / Third-party LLM":
            p = LLM_API_PRICING.get(selected_model, {})
            st.table(pd.DataFrame([
                ("Provider",       p.get("provider","—")),
                ("Input price",    f"${p.get('input_per_1m',0):.2f} / 1M tokens"),
                ("Output price",   f"${p.get('output_per_1m',0):.2f} / 1M tokens"),
                ("Context window", f"{p.get('context_window_k','?')}K tokens"),
                ("Quality score",  f"{p.get('quality_score','?')} / 10"),
            ], columns=["Attribute","Value"]).set_index("Attribute"))

    sub_hdr("Data Pipeline")
    d1, d2, d3 = st.columns(3)
    with d1:
        monthly_data_ingestion_gb = st.number_input("Monthly Data Ingestion (GB)", 1, 100_000, 200, key="ingest_gb")
        transformation_hours      = st.number_input("Monthly ETL Compute Hours",   1,  10_000, 100, key="etl_h")
    with d2:
        vector_db_vectors_m      = st.number_input("Vector DB Size (M Vectors)",   0.1, 1000.0, 5.0,  step=0.5, key="vdb_m")
        monthly_vector_queries_m = st.number_input("Monthly Vector Queries (M)",   0.1, 1000.0, 2.0,  step=0.1, key="vq_m")
    with d3:
        vector_db_type = st.selectbox("Vector DB", list(VECTOR_DB_PRICING.keys()), key="vdb_type")

    pipeline = compute_data_pipeline_cost(monthly_data_ingestion_gb, transformation_hours,
                                           vector_db_vectors_m, monthly_vector_queries_m, vector_db_type, cloud_provider)

    sub_hdr("Engineering & Labor")
    e1, e2 = st.columns(2)
    with e1:
        team_comp = {}
        for role in LABOR_RATES:
            team_comp[role] = st.number_input(role, 0, 20,
                1 if role in ("ML Engineer","Data Engineer") else 0, key=f"hc_{role}")
    with e2:
        labor_alloc_pct = st.slider("% Time Allocated to GenAI", 10, 100, 70, key="labor_alloc")
        st.caption("Reduce if team works across multiple projects")

    labor = compute_labor_cost(team_comp, labor_alloc_pct)

    sub_hdr("Maintenance & Operations")
    maintenance_pct = st.slider("Maintenance as % of Infra + Pipeline", 5, 30, 15, key="maint_pct")
    maintenance     = compute_maintenance_cost(infra["total"], pipeline["total"], maintenance_pct)

    tco = compute_total_tco(infra, model_api_cost, pipeline, labor, maintenance)

    sub_hdr("Monthly Cost Breakdown")
    breakdown = tco["breakdown"]
    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"]*len(breakdown) + ["total"],
        x=list(breakdown.keys()) + ["Total TCO"],
        y=list(breakdown.values()) + [tco["monthly"]],
        text=[fmt(v) for v in list(breakdown.values()) + [tco["monthly"]]],
        textposition="outside",
        textfont=dict(color=GC["text"], size=11, family="Roboto, Arial"),
        connector={"line": {"color": GC["border"], "width": 1, "dash": "dot"}},
        increasing={"marker": {"color": GC["blue"],  "line": {"color": GC["blue"],  "width": 0}}},
        totals=   {"marker": {"color": GC["green"], "line": {"color": GC["green"], "width": 0}}},
    ))
    fig_wf.update_layout(**chart_layout(height=400, yaxis_title="USD / Month", showlegend=False))
    st.plotly_chart(fig_wf, use_container_width=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        with st.expander("Infrastructure Detail"):
            st.table(pd.DataFrame([
                ("GPU Compute",    fmt(infra["gpu_compute"])),
                ("Storage",        fmt(infra["storage"])),
                ("Networking",     fmt(infra["networking"])),
                ("Cloud Overhead", fmt(infra["cloud_overhead"])),
            ], columns=["Component","Monthly"]).set_index("Component"))
    with col_b:
        with st.expander("Data Pipeline Detail"):
            st.table(pd.DataFrame([
                ("Ingestion",         fmt(pipeline["ingestion"])),
                ("ETL / Transform",   fmt(pipeline["transformation"])),
                ("Vector DB Storage", fmt(pipeline["vector_db_storage"])),
                ("Vector DB Queries", fmt(pipeline["vector_db_queries"])),
                ("Raw Storage",       fmt(pipeline["raw_storage"])),
            ], columns=["Component","Monthly"]).set_index("Component"))
    with col_c:
        with st.expander("Labor Detail"):
            rows = [(r, fmt(c)) for r, c in labor["breakdown"].items() if c > 0]
            if rows:
                st.table(pd.DataFrame(rows, columns=["Role","Monthly"]).set_index("Role"))


# ────────────────────────────────────────────────────────────────────────────
# CONTENT TAB 1 — Overview
# ────────────────────────────────────────────────────────────────────────────
with tab_ov:
    # Hero
    st.markdown(f"""
    <div class="gc-hero">
        <div class="gc-eyebrow">☁️ Cloud-agnostic GenAI cost intelligence</div>
        <h1>Understand the true cost and ROI of GenAI investments</h1>
        <p>
            GenAI cost is not only model pricing. The real total cost of ownership spans tokens,
            infrastructure, data pipelines, security, governance, operations, and business value
            realization — across any cloud, model provider, or deployment pattern.
        </p>
        <div class="gc-cta-row">
            <span class="gc-btn-primary">Open Calculator</span>
            <span class="gc-btn-secondary">View Scenarios</span>
        </div>
    </div>""", unsafe_allow_html=True)

    # Live KPI row
    cost_per_query = tco["monthly"] / monthly_queries if monthly_queries else 0
    k1, k2, k3, k4 = st.columns(4)
    with k1: gc_card("Monthly TCO",    fmt(tco["monthly"]),        f"Annual: {fmt(tco['annual'])}",        "gc-blue")
    with k2: gc_card("Cost Per Query", f"${cost_per_query:.5f}",   f"{monthly_queries:,} queries/month",   "gc-green")
    with k3: gc_card("Annual TCO",     fmt(tco["annual"]),         f"3-Year: {fmt(tco['three_year'])}",    "gc-yellow")
    with k4: gc_card("Top Cost Driver", max(tco["breakdown"], key=tco["breakdown"].get),
                     f"{fmt(tco['breakdown'][max(tco['breakdown'], key=tco['breakdown'].get)])}/month", "gc-red")

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # What is Tokenomics
    st.markdown("""
    <div class="gc-section-title">What is GenAI Tokenomics?</div>
    <div class="gc-section-sub">
        Tokenomics is the cost model behind all GenAI usage. Every prompt, response, embedding,
        retrieval step, agent action, and system instruction consumes tokens — and tokens drive cost.
        Understanding tokenomics is the foundation of any accurate TCO model.
    </div>""", unsafe_allow_html=True)

    st.markdown("""
    <div class="gc-callout">
        <div class="gc-callout-title">🪙 Why token costs are hard to estimate</div>
        <div class="gc-callout-body">
            Token usage changes based on prompt length, response length, number of active users,
            conversation depth, retrieval context size, model choice, agent loop steps, retry frequency,
            and application design. A small change in prompt engineering or model routing can create a
            large difference in monthly spend. Most early estimates undercount total cost by 2–4×
            because they only model the API cost and ignore infrastructure, data pipelines, security,
            governance, and operations.
        </div>
    </div>""", unsafe_allow_html=True)

    tok1, tok2, tok3, tok4 = st.columns(4)
    with tok1: st.markdown(info_card("📥", "Input Tokens",
        "Every prompt, document chunk, system instruction, conversation history, and retrieved context consumes input tokens billed at the input rate."), unsafe_allow_html=True)
    with tok2: st.markdown(info_card("📤", "Output Tokens",
        "Every response, generated code block, structured JSON, reasoning trace, and agent action output is billed at a higher output rate per token."), unsafe_allow_html=True)
    with tok3: st.markdown(info_card("🔢", "Embedding Tokens",
        "Document indexing, semantic search, reranking, and similarity matching each require embedding model calls — adding to per-query token cost."), unsafe_allow_html=True)
    with tok4: st.markdown(info_card("🔁", "Agent Loop Tokens",
        "Agentic workflows multiply token usage — one user request may trigger 5–20 model calls, each with its own prompt, tool call, and response."), unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # Six pillars
    st.markdown("""
    <div class="gc-section-title">Six Pillars of GenAI Total Cost of Ownership</div>
    <div class="gc-section-sub">
        The most accurate GenAI business case combines all six cost pillars. Token cost is the most
        visible line item, but infrastructure, data pipelines, security, and operations are often the
        largest long-term TCO drivers.
    </div>""", unsafe_allow_html=True)

    p1, p2, p3 = st.columns(3)
    with p1: st.markdown(info_card("🧠", "Model & Token Cost",
        "Input tokens, output tokens, embeddings, reranking, fine-tuning, batch inference jobs, agent calls, and evaluation runs. Price varies significantly across model tiers."), unsafe_allow_html=True)
    with p2: st.markdown(info_card("🖥️", "Infrastructure Cost",
        "GPU or CPU inference compute, hosted endpoints, containers, orchestration, auto-scaling, networking, storage, backup, and disaster recovery infrastructure."), unsafe_allow_html=True)
    with p3: st.markdown(info_card("🗄️", "Data & Retrieval Cost",
        "Vector databases, document ingestion pipelines, chunking, indexing, metadata storage, retrieval API calls, data refresh cycles, and embedding refresh."), unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    p4, p5, p6 = st.columns(3)
    with p4: st.markdown(info_card("🔒", "Security & Governance",
        "Prompt filtering, DLP controls, audit logging, access management, red teaming, policy enforcement, compliance evidence generation, and model risk management."), unsafe_allow_html=True)
    with p5: st.markdown(info_card("⚙️", "Operations & Support",
        "Monitoring, alerting, incident response, quality evaluation, prompt lifecycle management, release management, platform engineering, and end-user support."), unsafe_allow_html=True)
    with p6: st.markdown(info_card("📈", "Business Value",
        "Productivity gains, automation savings, customer support deflection, revenue uplift, faster decision cycles, reduced manual effort, and improved employee experience."), unsafe_allow_html=True)

    st.markdown("<div style='height:28px'></div>", unsafe_allow_html=True)

    # Eight challenges
    st.markdown("""
    <div class="gc-section-title">Why Accurate GenAI TCO Is Hard to Calculate</div>
    <div class="gc-section-sub">
        Most GenAI TCO estimates are wrong — not because the models are bad, but because usage
        patterns are dynamic, many costs are indirect, and cost attribution across teams is complex.
        These are the eight most common estimation blind spots.
    </div>""", unsafe_allow_html=True)

    ch1, ch2 = st.columns(2)
    with ch1:
        st.markdown(challenge_card("📊", "Unpredictable Usage Growth",
            "User adoption, request frequency, and conversation length change quickly after launch. A 50-user pilot can hit 10× the token volume when rolled out to 5,000 users."), unsafe_allow_html=True)
        st.markdown(challenge_card("🔀", "Variable Token Consumption",
            "Different prompts, document types, and response styles produce very different token volumes per request. A 20% increase in average context length can raise monthly cost by 40%."), unsafe_allow_html=True)
        st.markdown(challenge_card("🤖", "Hidden Agent & Tool Call Cost",
            "One business task may trigger 10–20 model calls, tool invocations, and retrieval steps internally — each billed at full token rates. Agent loops are the most underestimated cost driver."), unsafe_allow_html=True)
        st.markdown(challenge_card("💰", "Model Price Differences by Task",
            "Input rates, output rates, embedding rates, and batch pricing vary significantly across model tiers. Using the wrong model for routine tasks inflates cost by 5–10×."), unsafe_allow_html=True)
    with ch2:
        st.markdown(challenge_card("⚡", "GPU Utilization Sensitivity",
            "Infrastructure cost depends heavily on GPU utilization, request concurrency, latency targets, and auto-scaling thresholds. A 30% utilization difference can double per-request cost."), unsafe_allow_html=True)
        st.markdown(challenge_card("⚖️", "Quality vs. Cost Tradeoff",
            "The cheapest model or most compressed prompt may reduce answer quality, increase retry rates, trigger human review, or create downstream error costs that exceed the savings."), unsafe_allow_html=True)
        st.markdown(challenge_card("🔒", "Governance & Compliance Overhead",
            "Security controls, DLP filtering, audit logging, access management, and compliance evidence requirements add necessary enterprise cost rarely included in initial estimates."), unsafe_allow_html=True)
        st.markdown(challenge_card("📐", "ROI Attribution Complexity",
            "Proving that savings came directly from GenAI — rather than process, headcount, or seasonal changes — requires controlled baselines that most teams do not set up in advance."), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)

    # Live charts
    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("#### Live Cost Distribution")
        fig_pie = go.Figure(go.Pie(
            labels=list(tco["breakdown"].keys()),
            values=list(tco["breakdown"].values()),
            hole=0.52,
            marker=dict(colors=COLORS, line=dict(color="#FFFFFF", width=2)),
            textfont=dict(color=GC["text"], size=11, family="Roboto, Arial"),
            hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>",
        ))
        fig_pie.add_annotation(
            text=f"<b>{fmt(tco['monthly'])}</b><br><span style='font-size:11px;color:#5F6368'>per month</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=16, color=GC["text"]), align="center")
        fig_pie.update_layout(**chart_layout(height=340, showlegend=True,
                                              legend=dict(orientation="v", x=1.02, y=0.5)))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        st.markdown("#### 3-Year TCO Projection")
        proj    = project_costs_over_time(tco["monthly"], 36)
        df_proj = pd.DataFrame(proj)
        fig_tco = go.Figure()
        fig_tco.add_trace(go.Bar(
            x=df_proj["month"], y=df_proj["monthly"], name="Monthly Cost",
            marker_color="rgba(26,115,232,0.25)", yaxis="y2"))
        fig_tco.add_trace(go.Scatter(
            x=df_proj["month"], y=df_proj["cumulative"], name="Cumulative Cost",
            line=dict(color=GC["blue"], width=2.5),
            fill="tozeroy", fillcolor="rgba(26,115,232,0.07)"))
        layout = chart_layout(height=340)
        layout["yaxis"]  = dict(**layout.get("yaxis",{}), title="Cumulative ($)")
        layout["yaxis2"] = dict(title="Monthly ($)", overlaying="y", side="right",
                                 gridcolor="#F1F3F4", linecolor=GC["border"])
        layout["legend"] = dict(orientation="h", y=-0.18, bgcolor="rgba(255,255,255,0)")
        fig_tco.update_layout(**layout)
        st.plotly_chart(fig_tco, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# CONTENT TAB 2 — Scenarios
# ────────────────────────────────────────────────────────────────────────────
with tab_sc_info:
    section("Scenarios",
            "Compare GenAI deployment patterns before committing investment. Different architectures produce very different cost and ROI profiles.")

    st.markdown("""
    <div class="gc-callout">
        <div class="gc-callout-title">📐 How to choose a scenario</div>
        <div class="gc-callout-body">
            The best scenario balances cost, performance, quality, latency, security, compliance, and
            business value. A low-cost model may produce poor responses, while an expensive model may
            be unnecessary for simple classification or retrieval tasks. Evaluate using total value per
            query — not just cost per query. Use the Scenario Comparison calculator tab to run numbers.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    sc1, sc2 = st.columns(2)
    with sc1:
        st.markdown("""
        <div class="gc-scenario-card">
            <span class="gc-scenario-badge" style="background:#E8F0FE;color:#174EA6;">Baseline</span>
            <div class="gc-scenario-title">Cloud API — Direct</div>
            <div class="gc-scenario-desc">
                Standard model API usage with default prompts, no caching, and basic observability.
                Fastest path from idea to production. Best for pilots, proof-of-concept, and early
                exploration where speed matters more than cost efficiency.
            </div>
            <div class="gc-tag-row">
                <span class="gc-tag">No infra overhead</span>
                <span class="gc-tag">Pay-per-token</span>
                <span class="gc-tag">Fast to deploy</span>
                <span class="gc-tag">High token cost at scale</span>
                <span class="gc-tag">Zero MLOps burden</span>
            </div>
        </div>""", unsafe_allow_html=True)
    with sc2:
        st.markdown("""
        <div class="gc-scenario-card">
            <span class="gc-scenario-badge" style="background:#E6F4EA;color:#137333;">Optimized</span>
            <div class="gc-scenario-title">Cloud API — Cost Optimized</div>
            <div class="gc-scenario-desc">
                Cost-optimized design using prompt compression, semantic caching, model routing, and
                reduced context size. Applies 40–60% token savings over baseline. Best for production
                workloads scaling to tens of thousands of users.
            </div>
            <div class="gc-tag-row">
                <span class="gc-tag">40–60% token savings</span>
                <span class="gc-tag">Model routing</span>
                <span class="gc-tag">Semantic caching</span>
                <span class="gc-tag">Batch processing</span>
                <span class="gc-tag">Context compression</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    sc3, sc4 = st.columns(2)
    with sc3:
        st.markdown("""
        <div class="gc-scenario-card">
            <span class="gc-scenario-badge" style="background:#FEF7E0;color:#B06000;">Enterprise</span>
            <div class="gc-scenario-title">Fine-tuned Open Source</div>
            <div class="gc-scenario-desc">
                Domain-adapted open model running on dedicated cloud GPU. One-time fine-tuning
                investment with lower ongoing inference cost at scale. Best for specialized tasks
                exceeding 1M queries per month where domain accuracy matters.
            </div>
            <div class="gc-tag-row">
                <span class="gc-tag">Training CapEx</span>
                <span class="gc-tag">GPU inference cost</span>
                <span class="gc-tag">Domain accuracy</span>
                <span class="gc-tag">MLOps overhead</span>
                <span class="gc-tag">Predictable cost</span>
            </div>
        </div>""", unsafe_allow_html=True)
    with sc4:
        st.markdown("""
        <div class="gc-scenario-card">
            <span class="gc-scenario-badge" style="background:#FCE8E6;color:#C5221F;">Self-hosted</span>
            <div class="gc-scenario-title">Self-hosted / Private Cluster</div>
            <div class="gc-scenario-desc">
                Full GPU cluster with complete data sovereignty. No token costs. Every query runs on
                owned infrastructure. Best for regulated industries, very high query volume, and
                organizations with strict data residency requirements.
            </div>
            <div class="gc-tag-row">
                <span class="gc-tag">Full data control</span>
                <span class="gc-tag">Data sovereignty</span>
                <span class="gc-tag">High CapEx</span>
                <span class="gc-tag">GPU ops team required</span>
                <span class="gc-tag">No token billing</span>
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    sub_hdr("Scenario Selection Guide")

    guide_data = {
        "Criterion": [
            "Query volume", "Data sensitivity", "Budget type",
            "Team size", "Time to production", "Domain accuracy need",
            "Cost predictability", "Compliance requirements"
        ],
        "Cloud API (Baseline)": [
            "< 500K / month", "Low–Medium", "OpEx preferred",
            "Small (1–3)", "Days", "General purpose OK",
            "Variable (usage-based)", "Low–Medium"
        ],
        "Optimized API": [
            "500K–5M / month", "Low–Medium", "OpEx preferred",
            "Medium (2–5)", "Weeks", "General purpose OK",
            "Moderate (with caching)", "Low–Medium"
        ],
        "Fine-tuned OSS": [
            "> 1M / month", "Medium–High", "Mixed CapEx + OpEx",
            "Medium–Large (5–15)", "1–3 months", "High",
            "High (fixed GPU cost)", "Medium"
        ],
        "Self-hosted": [
            "> 5M / month", "Very High", "CapEx heavy",
            "Large (10+)", "3–6 months", "Full control",
            "Very High (fixed infra)", "Very High"
        ],
    }
    st.dataframe(pd.DataFrame(guide_data).set_index("Criterion"), use_container_width=True)

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    st.markdown("""
    <div style="font-size:13px;color:#5F6368;line-height:1.7;padding:16px 20px;
         background:#F8FAFD;border:1px solid #DADCE0;border-radius:12px;">
        <strong style="color:#202124">Key insight:</strong> Most enterprises start with Cloud API (Baseline) for speed,
        optimize to Cloud API (Optimized) within 90 days of production, and evaluate Fine-tuned OSS
        at 6–12 months when usage volume justifies the training investment.
        Self-hosted is typically a 12–24 month journey requiring a dedicated platform team.
    </div>""", unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# CONTENT TAB 3 — Optimization
# ────────────────────────────────────────────────────────────────────────────
with tab_opt_info:
    section("Optimization",
            "Reduce GenAI cost without reducing business value. Apply these six levers to lower token waste, improve model selection, and maximize value per request.")

    st.markdown("""
    <div class="gc-callout">
        <div class="gc-callout-title">💡 The goal of GenAI optimization</div>
        <div class="gc-callout-body">
            The best GenAI cost strategy combines token reduction, model routing, caching, retrieval
            tuning, GPU utilization improvement, and strong governance. The goal is not to minimize
            cost at all times, but to maximize value per token. Use the GPU Optimization calculator
            tab and Cost Components tab to quantify savings from each technique.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    ot1, ot2, ot3 = st.columns(3)
    with ot1:
        st.markdown(opt_card("🪙", "Token Optimization",
            "Control prompt size, response length, context injection, and conversation history to reduce token waste. The highest-leverage, lowest-effort cost reduction available.",
            ["Shorten and compress system prompts",
             "Limit retrieved context to relevant chunks only",
             "Summarize long conversation history before injecting",
             "Use structured outputs (JSON) to reduce verbosity",
             "Remove duplicate or boilerplate context",
             "Track input-to-output token ratio monthly"]), unsafe_allow_html=True)
    with ot2:
        st.markdown(opt_card("🔀", "Model Routing",
            "Route tasks to the right model tier. Not every request needs the most powerful or most expensive model. Routing saves 40–60% on most enterprise workloads.",
            ["Simple FAQ or lookup → small / economy model",
             "Summarization → mid-tier model",
             "Complex reasoning → advanced model",
             "Bulk classification → batch or economy tier",
             "Sensitive or high-risk → governed model path",
             "Track cost-per-task by routing tier"]), unsafe_allow_html=True)
    with ot3:
        st.markdown(opt_card("💾", "Caching Strategy",
            "Repeated questions, common documents, and similar prompts can be cached to avoid redundant model calls. Typical cache-hit rates of 20–40% for support use cases.",
            ["Exact-match prompt caching",
             "Semantic similarity caching for near-duplicates",
             "Embedding result caching",
             "Retrieval result caching for common queries",
             "Frequently-asked-question response cache",
             "Cache-hit rate tracking and monthly review"]), unsafe_allow_html=True)

    st.markdown("<div style='height:14px'></div>", unsafe_allow_html=True)
    ot4, ot5, ot6 = st.columns(3)
    with ot4:
        st.markdown(opt_card("🔍", "Retrieval Tuning",
            "RAG improves accuracy, but over-retrieval inflates token cost. Optimize chunking, ranking, and context selection to pass only what the model needs.",
            ["Right-size chunk length for your content type",
             "Use metadata filters before vector search",
             "Limit top-k retrieval to 3–5 documents max",
             "Apply reranking selectively, not globally",
             "Use document summaries instead of full text",
             "Track retrieval-to-answer conversion quality"]), unsafe_allow_html=True)
    with ot5:
        st.markdown(opt_card("🖥️", "GPU & Infra Efficiency",
            "For self-hosted or fine-tuned models, GPU utilization is the largest cost driver. Low utilization means expensive compute sitting idle.",
            ["Target 65–80% GPU utilization",
             "Use quantization (INT8 / FP8) where quality allows",
             "Separate real-time and batch inference workloads",
             "Auto-scale on request queue depth",
             "Use reserved capacity for predictable base load",
             "Track cost per 1,000 inference requests"]), unsafe_allow_html=True)
    with ot6:
        st.markdown(opt_card("🤖", "Agent Workflow Control",
            "Agentic systems create hidden cost — one user request can trigger many model calls, tool calls, and retries. Each is billed at full token rates.",
            ["Limit maximum agent reasoning steps",
             "Add deterministic stop conditions",
             "Track model calls and tool calls per task",
             "Avoid redundant tool invocations",
             "Cache intermediate agent results",
             "Use deterministic logic where LLM is not needed"]), unsafe_allow_html=True)

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)
    sub_hdr("Optimization Priority Matrix")
    matrix_data = {
        "Technique": [
            "Model routing", "Token / prompt compression", "Semantic caching",
            "Retrieval tuning", "GPU utilization", "Agent step limits",
            "Reserved capacity", "Quantization"
        ],
        "Effort": ["Low","Low","Medium","Medium","High","Low","Low","High"],
        "Impact": ["Very High","High","High","Medium","Very High","Medium","High","Medium"],
        "Typical Savings": ["40–60%","15–30%","20–40%","10–20%","20–50%","10–25%","35–45%","15–30%"],
        "Applies To": [
            "All API deployments", "All deployments", "High-volume support",
            "RAG workloads", "Self-hosted / fine-tuned", "Agentic workflows",
            "Predictable GPU load", "Self-hosted inference"
        ],
    }
    st.dataframe(pd.DataFrame(matrix_data), use_container_width=True, hide_index=True)


# ────────────────────────────────────────────────────────────────────────────
# CONTENT TAB 4 — Reports
# ────────────────────────────────────────────────────────────────────────────
with tab_rep_info:
    section("Reports",
            "Generate executive-ready GenAI TCO and ROI reports. Use the ROI & Payback and Docs & Export calculator tabs to produce and download these reports.")

    st.markdown("""
    <div class="gc-callout">
        <div class="gc-callout-title">📋 Report disclaimer</div>
        <div class="gc-callout-body">
            All values are estimates based on provided assumptions. Actual GenAI cost may vary based
            on user behavior, model pricing changes, token volume growth, infrastructure utilization,
            data size, latency requirements, governance controls, and support needs.
            Use these reports as directional guidance for investment decisions, not as fixed commitments.
        </div>
    </div>""", unsafe_allow_html=True)

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    r1, r2, r3 = st.columns(3)
    with r1:
        st.markdown(report_card("1", "Executive Summary",
            "A business-level summary of estimated monthly TCO, annual TCO, savings potential, ROI, payback period, and top cost drivers — formatted for CFO and board presentations.",
            ["Monthly and annual TCO",
             "Estimated savings vs. alternatives",
             "ROI percentage and payback period",
             "Top 3 cost drivers",
             "Recommended optimization actions"]), unsafe_allow_html=True)
    with r2:
        st.markdown(report_card("2", "Tokenomics Summary",
            "Explains how token usage contributes to cost across prompts, responses, embeddings, retrieval, agent calls, and evaluation workloads — by use case.",
            ["Monthly input and output token volume",
             "Average tokens per request",
             "Average requests per active user",
             "Cost per 1,000 requests",
             "Cost per active user per month",
             "Token cost breakdown by use case"]), unsafe_allow_html=True)
    with r3:
        st.markdown(report_card("3", "Cost Breakdown Report",
            "Shows how total cost is distributed across model usage, infrastructure, storage, security, governance, monitoring, and operations — with trend projections.",
            ["Model / API cost",
             "GPU and compute cost",
             "Vector database and storage cost",
             "Data pipeline and ingestion cost",
             "Security and governance overhead",
             "Monitoring and operations cost"]), unsafe_allow_html=True)

    st.markdown("<div style='height:12px'></div>", unsafe_allow_html=True)
    r4, r5, r6 = st.columns(3)
    with r4:
        st.markdown(report_card("4", "Scenario Comparison",
            "Compares baseline, optimized, enterprise-secure, and self-hosted scenarios using consistent assumptions — so stakeholders can compare apples to apples.",
            ["Scenario assumptions documented",
             "Monthly and annual TCO by scenario",
             "Savings difference between scenarios",
             "Risk and compliance level comparison",
             "Operational complexity rating",
             "Recommended scenario with rationale"]), unsafe_allow_html=True)
    with r5:
        st.markdown(report_card("5", "ROI & Payback Report",
            "Connects GenAI investment to measurable business outcomes — productivity improvement, automation savings, customer support deflection, and reduced manual effort.",
            ["Annual benefit and annual cost",
             "Net savings and ROI percentage",
             "Payback period in months",
             "5-year NPV at 10% discount rate",
             "IRR calculation",
             "Benefit ramp and sensitivity analysis"]), unsafe_allow_html=True)
    with r6:
        st.markdown(report_card("6", "Assumptions & Methodology",
            "Documents all assumptions used in the TCO model so that finance, engineering, security, and business teams can review and challenge the calculation.",
            ["User count and request volume assumptions",
             "Token consumption assumptions",
             "Model and infrastructure assumptions",
             "Security and compliance cost assumptions",
             "Operations and support assumptions",
             "Business benefit methodology"]), unsafe_allow_html=True)

    st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
    sub_hdr("Available Export Formats")
    ef1, ef2, ef3 = st.columns(3)
    with ef1:
        st.markdown(info_card("📄", "Executive CSV",
            "Current scenario with all TCO, ROI, and benefit metrics in a single downloadable spreadsheet. Available in the Docs & Export calculator tab."), unsafe_allow_html=True)
    with ef2:
        st.markdown(info_card("📊", "Scenario Comparison CSV",
            "Side-by-side comparison of all four deployment scenarios with monthly cost, annual cost, and vs-current delta. Available in the Docs & Export calculator tab."), unsafe_allow_html=True)
    with ef3:
        st.markdown(info_card("📋", "Assumptions Reference",
            "Full formula reference, GPU pricing table, LLM API pricing table, and architecture decision guide. Available in the Docs & Export calculator tab."), unsafe_allow_html=True)


# ────────────────────────────────────────────────────────────────────────────
# CALCULATOR TAB 1 — Executive Dashboard
# ────────────────────────────────────────────────────────────────────────────
with tab_dashboard:
    st.markdown(f"""
    <div class="gc-hero">
        <div class="gc-eyebrow">☁️ Cloud cost intelligence for GenAI</div>
        <h1>Estimate and optimize the total cost of GenAI workloads</h1>
        <p>
            Build an executive-ready TCO model for foundation models, GPU infrastructure,
            token consumption, data pipelines, operations, and ROI — across GCP, AWS, and Azure.
        </p>
        <div class="gc-cta-row">
            <span class="gc-btn-primary">Start calculation</span>
            <span class="gc-btn-secondary">Export report</span>
        </div>
    </div>""", unsafe_allow_html=True)

    cost_per_query = tco["monthly"] / monthly_queries if monthly_queries else 0
    k1, k2, k3, k4 = st.columns(4)
    with k1: gc_card("Monthly TCO",    fmt(tco["monthly"]),             f"Annual: {fmt(tco['annual'])}",         "gc-blue")
    with k2: gc_card("Cost Per Query", f"${cost_per_query:.5f}",        f"{monthly_queries:,} queries/month",    "gc-green")
    with k3: gc_card("Annual TCO",     fmt(tco["annual"]),              f"3-Year: {fmt(tco['three_year'])}",     "gc-yellow")
    with k4: gc_card("Model Approach", model_approach.split(" ")[0],    deployment_type,                          "gc-red")

    st.markdown("<div style='height:24px'></div>", unsafe_allow_html=True)

    col_l, col_r = st.columns([1, 1])
    with col_l:
        st.markdown("#### Cost Distribution")
        fig_pie = go.Figure(go.Pie(
            labels=list(tco["breakdown"].keys()),
            values=list(tco["breakdown"].values()),
            hole=0.52,
            marker=dict(colors=COLORS, line=dict(color="#FFFFFF", width=2)),
            textfont=dict(color=GC["text"], size=11, family="Roboto, Arial"),
            hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>",
        ))
        fig_pie.add_annotation(
            text=f"<b>{fmt(tco['monthly'])}</b><br><span style='font-size:11px;color:#5F6368'>per month</span>",
            x=0.5, y=0.5, showarrow=False, font=dict(size=16, color=GC["text"]), align="center")
        fig_pie.update_layout(**chart_layout(height=360, showlegend=True,
                                              legend=dict(orientation="v", x=1.02, y=0.5)))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        st.markdown("#### 3-Year Cumulative TCO")
        proj    = project_costs_over_time(tco["monthly"], 36)
        df_proj = pd.DataFrame(proj)
        fig_tco = go.Figure()
        fig_tco.add_trace(go.Bar(
            x=df_proj["month"], y=df_proj["monthly"], name="Monthly Cost",
            marker_color="rgba(26,115,232,0.25)", yaxis="y2"))
        fig_tco.add_trace(go.Scatter(
            x=df_proj["month"], y=df_proj["cumulative"], name="Cumulative Cost",
            line=dict(color=GC["blue"], width=2.5),
            fill="tozeroy", fillcolor="rgba(26,115,232,0.07)"))
        layout = chart_layout(height=360)
        layout["yaxis"]  = dict(**layout.get("yaxis",{}), title="Cumulative ($)")
        layout["yaxis2"] = dict(title="Monthly ($)", overlaying="y", side="right",
                                 gridcolor="#F1F3F4", linecolor=GC["border"])
        layout["legend"] = dict(orientation="h", y=-0.18, bgcolor="rgba(255,255,255,0)")
        fig_tco.update_layout(**layout)
        st.plotly_chart(fig_tco, use_container_width=True)

    st.markdown("#### Cost Summary")
    st.dataframe(pd.DataFrame({
        "Period":      ["Monthly","Quarterly","Annual","3-Year","5-Year"],
        "TCO":         [fmt(tco["monthly"]), fmt(tco["monthly"]*3), fmt(tco["annual"]),
                        fmt(tco["three_year"]), fmt(tco["five_year"])],
        "$/Query":     [f"${tco['monthly']/monthly_queries:.5f}"]*5 if monthly_queries else ["—"]*5,
        "Top Driver":  [max(tco["breakdown"], key=tco["breakdown"].get)]*5,
    }), use_container_width=True, hide_index=True)

    st.markdown("#### Strategic Recommendations")
    largest = max(tco["breakdown"], key=tco["breakdown"].get)
    if largest == "Labor":
        rec("👥", "Labor Dominates Costs",
            "Consider managed MLOps services to reduce engineering overhead. Evaluate platforms like Vertex AI that bundle infra + tooling.",
            GC["blue"])
    if largest == "Model / API":
        rec("💡", "Optimize Model Selection",
            "API spend is your top cost driver. Route routine queries to Claude Haiku or Gemini Flash and reserve premium models for complex tasks — typical savings: 40–60%.",
            GC["green"])
    if infra["gpu_compute"] > 0 and utilization_pct < 40:
        rec("🖥️", f"Low GPU Utilization ({utilization_pct}%)",
            "Target 70%+ utilization. Use preemptible instances for batch workloads and auto-scaling for variable inference demand.",
            GC["yellow"])
    if deployment_type == "Cloud API (Managed)":
        rec("☁️", "Evaluate Reserved Pricing",
            "Committing to 1-year reserved GPU instances typically saves 35–45% vs on-demand at your current usage scale.",
            GC["red"])
    rec("📊", "Monitor & Iterate",
        "Set monthly cost budget alerts and track cost-per-query trends. A 10% token reduction at 500K queries/month yields measurable annual savings.",
        GC["purple"])


# ────────────────────────────────────────────────────────────────────────────
# CALCULATOR TAB 3 — Scenario Comparison
# ────────────────────────────────────────────────────────────────────────────
with tab_scenario:
    section("Scenario Comparison",
            "Compare baseline, optimized, and alternative deployment options side by side.")

    base_inputs = dict(monthly_queries=monthly_queries, avg_input_tokens=avg_input_tokens,
                       avg_output_tokens=avg_output_tokens, storage_gb=storage_gb,
                       networking_egress_gb=networking_egress_gb, utilization_pct=utilization_pct)
    scenarios = compute_scenario_comparison(base_inputs)

    s_cols = st.columns(len(scenarios))
    colors_sc = ["gc-blue","gc-green","gc-red","gc-yellow"]
    lowest = min(scenarios, key=lambda k: scenarios[k]["total"])
    for i, (name, vals) in enumerate(scenarios.items()):
        with s_cols[i]:
            badge = " ✅" if name == lowest else ""
            gc_card(name.upper(), fmt(vals["total"]),
                    f"Model: {fmt(vals['model'])} · Infra: {fmt(vals['infra'])}{badge}",
                    colors_sc[i % len(colors_sc)])

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)

    fig_sc = go.Figure()
    for i, (name, vals) in enumerate(scenarios.items()):
        fig_sc.add_trace(go.Bar(
            name=name,
            x=["Model / Infra Cost","Infrastructure Cost","Total Monthly"],
            y=[vals["model"], vals["infra"], vals["total"]],
            marker=dict(color=COLORS[i], opacity=0.85),
            text=[fmt(vals["model"]), fmt(vals["infra"]), fmt(vals["total"])],
            textposition="outside",
            textfont=dict(color=GC["text"], size=10),
        ))
    fig_sc.update_layout(**chart_layout(barmode="group", height=400, yaxis_title="USD / Month",
                                         legend=dict(orientation="h", y=-0.2)))
    st.plotly_chart(fig_sc, use_container_width=True)

    st.markdown("#### Detailed Breakdown")
    sc_rows = []
    for name, vals in scenarios.items():
        diff = tco["monthly"] - vals["total"]
        sc_rows.append({"Scenario": name, "Model Cost": fmt(vals["model"]),
                     "Infra Cost": fmt(vals["infra"]), "Total/Month": fmt(vals["total"]),
                     "Annual": fmt(vals["total"]*12),
                     "vs Current": ("+"+fmt(diff) if diff > 0 else fmt(diff)),
                     "": ("✅ Best" if name == lowest else "")})
    st.dataframe(pd.DataFrame(sc_rows), use_container_width=True, hide_index=True)

    st.divider()
    sub_hdr("Model-by-Model API Cost Comparison")
    st.caption(f"{monthly_queries:,} queries/month · {avg_input_tokens} input + {avg_output_tokens} output tokens")

    model_rows = []
    for mname, mp in LLM_API_PRICING.items():
        cost = compute_api_model_cost(mname, monthly_queries, avg_input_tokens, avg_output_tokens)
        model_rows.append({"Model": mname, "Provider": mp["provider"],
                           "Monthly": fmt(cost), "Annual": fmt(cost*12),
                           "Input $/1M": f"${mp['input_per_1m']}", "Output $/1M": f"${mp['output_per_1m']}",
                           "Context": f"{mp['context_window_k']}K",
                           "Quality": "⭐"*mp["quality_score"], "_s": cost})
    st.dataframe(pd.DataFrame(model_rows).sort_values("_s").drop(columns=["_s"]),
                 use_container_width=True, hide_index=True)

    st.divider()
    sub_hdr("3-Year Cloud vs On-Prem Projection")
    cloud_m  = list(scenarios.values())[0]["total"] + labor["total"] + pipeline["total"]
    onprem_m = list(scenarios.values())[2]["total"] + labor["total"] + pipeline["total"]
    capex    = GPU_PRICING.get(gpu_type,{}).get("on_demand_hourly",3.67)*gpu_count*8760*0.2
    months_r   = list(range(1, 37))
    cloud_cum  = [cloud_m*m for m in months_r]
    onprem_cum = [capex+onprem_m*m for m in months_r]
    fig_cvo = go.Figure()
    fig_cvo.add_trace(go.Scatter(x=months_r, y=cloud_cum, name="Cloud API",
                                  line=dict(color=GC["blue"], width=2.5),
                                  fill="tozeroy", fillcolor="rgba(26,115,232,0.07)"))
    fig_cvo.add_trace(go.Scatter(x=months_r, y=onprem_cum, name="On-Prem (incl. CapEx)",
                                  line=dict(color=GC["red"], width=2.5),
                                  fill="tozeroy", fillcolor="rgba(234,67,53,0.05)"))
    cross = next((m for m,(c,o) in enumerate(zip(cloud_cum,onprem_cum),1) if c > o), None)
    if cross:
        fig_cvo.add_vline(x=cross, line_dash="dash", line_color=GC["green"],
                          annotation_text=f"Crossover: Month {cross}",
                          annotation_font=dict(color=GC["green"], size=11))
    fig_cvo.update_layout(**chart_layout(height=360, yaxis_title="Cumulative Cost ($)", xaxis_title="Month"))
    st.plotly_chart(fig_cvo, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# CALCULATOR TAB 4 — GPU Optimization
# ────────────────────────────────────────────────────────────────────────────
with tab_gpu:
    section("GPU Optimization",
            "Estimate GPU cost, utilization curves, reserved capacity savings, and idle waste.")

    g1, g2 = st.columns(2)
    with g1:
        gpu_opt_type  = st.selectbox("Select GPU for Analysis", list(GPU_PRICING.keys()), key="gpu_opt")
        gpu_opt_count = st.slider("GPU Count", 1, 32, 4, key="gpu_opt_count")
    with g2:
        util_range = st.slider("Utilization Range (%)", 10, 100, (20,90), key="util_range")

    gi = GPU_PRICING[gpu_opt_type]

    sp1, sp2, sp3, sp4 = st.columns(4)
    sp1.metric("Memory",          f"{gi['memory_gb']} GB")
    sp2.metric("BF16 Performance",f"{gi['flops_bf16_tflops']} TFLOPS")
    sp3.metric("On-Demand Rate",  f"${gi['on_demand_hourly']:.2f}/hr")
    sp4.metric("Reserved 1yr",    f"${gi['reserved_1yr_hourly']:.2f}/hr",
               f"-{round((1-gi['reserved_1yr_hourly']/gi['on_demand_hourly'])*100)}%")

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)

    util_pts = list(range(10, 101, 5))
    data = []
    for tier, label in [("on_demand","On-Demand"),("reserved_1yr","Reserved 1yr"),("reserved_3yr","Reserved 3yr")]:
        for u in util_pts:
            data.append({"Utilization (%)": u,
                          "Monthly Cost ($)": compute_gpu_monthly_cost(gpu_opt_type, gpu_opt_count, u, tier),
                          "Tier": label})
    df_util = pd.DataFrame(data)
    fig_util = px.line(df_util, x="Utilization (%)", y="Monthly Cost ($)", color="Tier",
                       color_discrete_sequence=[GC["red"], GC["blue"], GC["green"]], markers=True)
    fig_util.add_vrect(x0=util_range[0], x1=util_range[1],
                       fillcolor="rgba(26,115,232,0.06)", line_width=0,
                       annotation_text="Your range",
                       annotation_font=dict(color=GC["blue"], size=10))
    fig_util.update_layout(**chart_layout(height=380, yaxis_title="Monthly Cost ($)"))
    st.plotly_chart(fig_util, use_container_width=True)

    sub_hdr("All GPU Types — Side-by-Side")
    gpu_rows = []
    for gname, gdata in GPU_PRICING.items():
        od = compute_gpu_monthly_cost(gname, gpu_opt_count, utilization_pct, "on_demand")
        r1 = compute_gpu_monthly_cost(gname, gpu_opt_count, utilization_pct, "reserved_1yr")
        r3 = compute_gpu_monthly_cost(gname, gpu_opt_count, utilization_pct, "reserved_3yr")
        gpu_rows.append({"GPU": gname, "Memory": f"{gdata['memory_gb']} GB",
                          "BF16 TFLOPS": gdata["flops_bf16_tflops"],
                          "On-Demand/mo": fmt(od), "Reserved 1yr/mo": fmt(r1), "Reserved 3yr/mo": fmt(r3),
                          "1yr Savings": f"{round((1-r1/od)*100) if od else 0}%",
                          "TFLOPS/$": f"{gdata['flops_bf16_tflops']/gdata['on_demand_hourly']:.1f}"})
    st.dataframe(pd.DataFrame(gpu_rows), use_container_width=True, hide_index=True)

    sub_hdr("Impact of Idle GPU Time")
    idle_data = [{"Utilization (%)": u,
                   "Active Cost ($)": compute_gpu_monthly_cost(gpu_opt_type, gpu_opt_count, u, "on_demand"),
                   "Idle Waste ($)":  compute_gpu_monthly_cost(gpu_opt_type, gpu_opt_count, u, "on_demand")*(1-u/100)*0.5}
                 for u in range(10, 101, 10)]
    df_idle = pd.DataFrame(idle_data)
    fig_idle = go.Figure()
    fig_idle.add_trace(go.Bar(x=df_idle["Utilization (%)"], y=df_idle["Active Cost ($)"],
                               name="Active Cost", marker_color=GC["blue"], opacity=0.8))
    fig_idle.add_trace(go.Bar(x=df_idle["Utilization (%)"], y=df_idle["Idle Waste ($)"],
                               name="Idle Waste", marker_color=GC["red"], opacity=0.75))
    fig_idle.update_layout(**chart_layout(barmode="overlay", height=320, yaxis_title="Monthly Cost ($)"))
    st.plotly_chart(fig_idle, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# CALCULATOR TAB 5 — ROI & Payback
# ────────────────────────────────────────────────────────────────────────────
with tab_roi:
    section("ROI & Business Case",
            "Quantify productivity gains, cost avoidance, revenue impact, payback period, and NPV.")

    sub_hdr("Legacy System Baseline")
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        legacy_type    = st.selectbox("Legacy System Type", list(LEGACY_SYSTEM_BENCHMARKS.keys()), key="legacy_type")
        legacy_units   = st.number_input("Units (agents / seats / licenses)", 1, 10_000, 50, key="legacy_units")
        legacy_monthly = LEGACY_SYSTEM_BENCHMARKS[legacy_type] * legacy_units / 12
        st.metric("Estimated Legacy Monthly Cost", fmt(legacy_monthly))
    with r1c2:
        initial_investment = st.number_input("Initial Setup Investment ($)", 0, 5_000_000, 50_000, step=5000, key="initial_inv")
        st.caption("Integration · data prep · training · infra setup · change management")

    sub_hdr("Productivity Gains")
    p1, p2, p3, p4 = st.columns(4)
    with p1: employees_affected  = st.number_input("Employees Affected", 1, 10_000, 100, key="emp_aff")
    with p2: hours_saved_monthly = st.slider("Hours Saved / Employee / Month", 0, 80, 15, key="hrs_saved")
    with p3: avg_hourly_rate     = st.number_input("Avg Hourly Rate ($)", 10, 200, 45, key="hrly_rate")
    with p4: automation_rate_pct = st.slider("Automation Rate %", 0, 100, 40, key="auto_rate")
    prod_gains = compute_productivity_gains(employees_affected, hours_saved_monthly, avg_hourly_rate, automation_rate_pct)

    sub_hdr("Cost Avoidance")
    ca1, ca2, ca3 = st.columns(3)
    with ca1:
        headcount_reduction = st.number_input("Headcount Reduction (FTEs)", 0, 500, 10, key="hc_red")
        avg_agent_salary    = st.number_input("Avg Agent Annual Salary ($)", 20_000, 200_000, 65_000, step=5000, key="agent_sal")
    with ca2:
        error_reduction_pct = st.slider("Error Reduction %", 0, 100, 60, key="err_red")
        avg_error_cost      = st.number_input("Avg Cost Per Error ($)", 0, 10_000, 200, key="err_cost")
    with ca3:
        monthly_error_volume = st.number_input("Monthly Error Volume", 0, 100_000, 500, key="err_vol")
    cost_avoid = compute_cost_avoidance(legacy_monthly, headcount_reduction, avg_agent_salary,
                                        error_reduction_pct, avg_error_cost, monthly_error_volume)

    sub_hdr("Revenue Impact")
    rv1, rv2, rv3 = st.columns(3)
    with rv1:
        monthly_interactions = st.number_input("Monthly Customer Interactions", 100, 10_000_000, 50_000, step=1000, key="cust_int")
        avg_rev_per_customer = st.number_input("Avg Revenue Per Customer ($)", 1, 10_000, 120, key="avg_rev")
    with rv2:
        baseline_csat = st.slider("Baseline CSAT Score", 1, 10, 6, key="base_csat")
        target_csat   = st.slider("Target CSAT Score",   1, 10, 8, key="tgt_csat")
    with rv3:
        churn_reduction_pct = st.slider("Churn Reduction %", 0, 50, 10, key="churn_red")
        upsell_lift_pct     = st.slider("Upsell Lift %",     0, 20,  5, key="upsell_lift")
    rev_impact = compute_revenue_impact(monthly_interactions, baseline_csat, target_csat,
                                        avg_rev_per_customer, churn_reduction_pct, upsell_lift_pct)

    roi_result  = compute_full_roi(tco["monthly"], prod_gains, cost_avoid, rev_impact, initial_investment, months=60)
    payback     = roi_result["payback_months"]
    payback_str = f"{payback:.1f} months" if payback != float("inf") else "Never"

    st.markdown("<div style='height:8px'></div>", unsafe_allow_html=True)
    kp1, kp2, kp3, kp4 = st.columns(4)
    with kp1: gc_card("Payback Period",    payback_str,                     "Months to break even",       "gc-blue")
    with kp2: gc_card("Monthly Net Benefit",fmt(roi_result["monthly_net"]), "Benefits minus TCO",         "gc-green" if roi_result["monthly_net"]>0 else "gc-red")
    with kp3: gc_card("3-Year NPV",        fmt(abs(roi_result["npv"]))+(" ✅" if roi_result["npv"]>0 else " ⚠️"), "At 10% discount rate", "gc-green" if roi_result["npv"]>0 else "gc-red")
    with kp4: gc_card("IRR",              f"{roi_result['irr_pct']:.1f}%" if roi_result["irr_pct"] else "—", "Internal Rate of Return", "gc-yellow")

    st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
    col_ben, col_time = st.columns(2)
    with col_ben:
        st.markdown("#### Monthly Benefits Breakdown")
        fig_ben = go.Figure(go.Pie(
            labels=list(roi_result["benefit_breakdown"].keys()),
            values=list(roi_result["benefit_breakdown"].values()),
            hole=0.5,
            marker=dict(colors=[GC["blue"],GC["green"],GC["purple"]], line=dict(color="#FFFFFF",width=2)),
            textfont=dict(color=GC["text"], size=11),
            hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>",
        ))
        total_ben = sum(roi_result["benefit_breakdown"].values())
        fig_ben.add_annotation(text=f"<b>{fmt(total_ben)}</b><br><span style='font-size:10px;color:#5F6368'>/ month</span>",
                               x=0.5, y=0.5, showarrow=False, font=dict(size=15, color=GC["text"]))
        fig_ben.update_layout(**chart_layout(height=340, showlegend=True,
                                              legend=dict(orientation="h", y=-0.12)))
        st.plotly_chart(fig_ben, use_container_width=True)

    with col_time:
        st.markdown("#### 5-Year Cumulative Value")
        df_tl = pd.DataFrame(roi_result["timeline"])
        fig_roi_chart = go.Figure()
        fig_roi_chart.add_trace(go.Scatter(
            x=df_tl["month"], y=df_tl["cumulative_benefit"], name="Cumulative Benefit",
            line=dict(color=GC["green"], width=2.5),
            fill="tozeroy", fillcolor="rgba(52,168,83,0.08)"))
        fig_roi_chart.add_trace(go.Scatter(
            x=df_tl["month"], y=df_tl["cumulative_cost"], name="Cumulative Cost",
            line=dict(color=GC["red"], width=2.5),
            fill="tozeroy", fillcolor="rgba(234,67,53,0.05)"))
        if payback != float("inf") and payback <= 60:
            fig_roi_chart.add_vline(x=payback, line_dash="dash", line_color=GC["yellow"],
                              annotation_text=f"Payback: Month {payback:.0f}",
                              annotation_font=dict(color=GC["blue"], size=10))
        fig_roi_chart.update_layout(**chart_layout(height=340, yaxis_title="Cumulative Value ($)",
                                                    xaxis_title="Month", legend=dict(orientation="h", y=-0.18)))
        st.plotly_chart(fig_roi_chart, use_container_width=True)

    st.markdown("#### ROI % Over Time")
    fig_rp = go.Figure()
    fig_rp.add_trace(go.Scatter(
        x=df_tl["month"], y=df_tl["roi_pct"],
        line=dict(color=GC["blue"], width=2.5),
        fill="tozeroy", fillcolor="rgba(26,115,232,0.07)",
        hovertemplate="Month %{x}<br>ROI: %{y:.1f}%<extra></extra>"))
    fig_rp.add_hline(y=0,   line_dash="dash",  line_color=GC["border"])
    fig_rp.add_hline(y=100, line_dash="dot",   line_color=GC["green"],
                     annotation_text="100% ROI",
                     annotation_font=dict(color=GC["green"], size=10),
                     annotation_position="top right")
    fig_rp.update_layout(**chart_layout(height=280, yaxis_title="ROI (%)", xaxis_title="Month"))
    st.plotly_chart(fig_rp, use_container_width=True)


# ────────────────────────────────────────────────────────────────────────────
# CALCULATOR TAB 6 — Docs & Export
# ────────────────────────────────────────────────────────────────────────────
with tab_export:
    section("Docs & Export",
            "Formula reference, pricing tables, architecture notes, and downloadable scenario outputs.")

    with st.expander("📐 Cost Formula Reference", expanded=True):
        st.markdown("""
**Infrastructure**
```
GPU Monthly = GPU_Count × 730hrs × Rate × Utilization%
            + GPU_Count × 730hrs × IdleRate × (1 − Utilization%)
Storage     = Storage_GB × Rate_per_GB_Month
Network     = Egress_GB  × Rate_per_GB
```
**Model / API**
```
API Cost = Queries × InputTokens  / 1,000,000 × InputRate
         + Queries × OutputTokens / 1,000,000 × OutputRate
```
**Data Pipeline**
```
Pipeline = Ingestion + ETL_Compute + VectorDB_Storage + VectorDB_Queries + Raw_Storage
```
**Total Monthly TCO**
```
TCO = Infrastructure + Model_API + Data_Pipeline + Labor + Maintenance
```
**ROI**
```
Monthly Net = Total_Monthly_Benefits − Monthly_TCO
Payback     = Initial_Investment ÷ Monthly_Net
NPV         = −CapEx + Σ (Net_Cashflow_t / (1 + r)^t)   [r = 10%]
IRR         = Bisection solve where NPV = 0
```""")

    with st.expander("💡 Live Pricing Reference"):
        st.markdown("#### GPU Rates (GCP / AWS / Azure)")
        st.dataframe(pd.DataFrame([{
            "GPU": k, "On-Demand $/hr": v["on_demand_hourly"],
            "Reserved 1yr $/hr": v["reserved_1yr_hourly"], "Reserved 3yr $/hr": v["reserved_3yr_hourly"],
            "Memory GB": v["memory_gb"], "BF16 TFLOPS": v["flops_bf16_tflops"],
        } for k, v in GPU_PRICING.items()]), use_container_width=True, hide_index=True)

        st.markdown("#### LLM API Pricing")
        st.dataframe(pd.DataFrame([{
            "Model": k, "Provider": v["provider"],
            "Input $/1M": v["input_per_1m"], "Output $/1M": v["output_per_1m"],
            "Context Window": f"{v['context_window_k']}K", "Quality": v["quality_score"],
        } for k, v in LLM_API_PRICING.items()]), use_container_width=True, hide_index=True)

    with st.expander("🏗️ Architecture Notes"):
        st.markdown("""
| Approach | Pros | Cons | Best For |
|---|---|---|---|
| **Cloud API** | Zero infra, instant scale | Token cost grows with volume | Early stage, <1M queries/mo |
| **Fine-tuned OSS** | Domain accuracy, lower long-term cost | Training CapEx, MLOps overhead | Specialized tasks, >1M queries/mo |
| **Self-hosted On-Prem** | Full control, data sovereignty | High CapEx, GPU ops team | Regulated industries, very high volume |

**Cost Optimization Levers**
1. **Semantic caching** — cache similar queries (reduces API calls 20–40%)
2. **Token compression** — concise system prompts, strip whitespace
3. **Model routing** — lightweight model for simple queries, premium for complex
4. **Batch processing** — group non-realtime requests for cheaper batch APIs
5. **Reserved capacity** — 1yr commit saves 35–45% vs on-demand
        """)

    with st.expander("📊 Export Current Scenario"):
        export = {
            "Company": company_name, "Industry": industry, "Use Case": use_case,
            "Monthly Queries": monthly_queries, "Cloud Provider": cloud_provider,
            "Deployment": deployment_type, "Model Approach": model_approach,
            "Monthly TCO ($)": round(tco["monthly"],2), "Annual TCO ($)": round(tco["annual"],2),
            "3-Year TCO ($)": round(tco["three_year"],2),
            "Infrastructure ($)": round(infra["total"],2), "Model/API ($)": round(model_api_cost,2),
            "Data Pipeline ($)": round(pipeline["total"],2), "Labor ($)": round(labor["total"],2),
            "Maintenance ($)": round(maintenance,2),
            "Monthly Benefits ($)": round(roi_result["monthly_benefits"],2),
            "Payback (months)": round(roi_result["payback_months"],1) if roi_result["payback_months"] != float("inf") else "N/A",
            "NPV ($)": round(roi_result["npv"],2),
        }
        df_exp = pd.DataFrame([export]).T.reset_index()
        df_exp.columns = ["Metric","Value"]
        st.dataframe(df_exp, use_container_width=True, hide_index=True)
        c_dl1, c_dl2 = st.columns(2)
        with c_dl1:
            st.download_button("⬇️  Download as CSV",
                df_exp.to_csv(index=False),
                f"genai_tco_{company_name.replace(' ','_')}.csv", "text/csv")
        with c_dl2:
            st.button("📄  Export PDF Report")
