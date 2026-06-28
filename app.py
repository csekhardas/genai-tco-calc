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
    page_title="Enterprise GenAI TCO & ROI Calculator",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Design tokens ───────────────────────────────────────────────────────────
G  = "#76b900"   # NVIDIA green
B  = "#4285F4"   # GCP blue
R  = "#ea4335"   # Google red
Y  = "#fbbc04"   # Google yellow
P  = "#a78bfa"   # purple
DARK  = "rgba(10,14,26,0)"
PLOT  = "rgba(13,27,42,0.5)"
GRID  = "rgba(255,255,255,0.05)"
FONT  = "#94a3b8"
COLORS = [G, B, R, Y, P, "#38bdf8"]

def chart_layout(**kw):
    base = dict(
        template="plotly_dark",
        paper_bgcolor=DARK,
        plot_bgcolor=PLOT,
        font=dict(family="Inter, 'Google Sans', sans-serif", color=FONT, size=12),
        xaxis=dict(gridcolor=GRID, zerolinecolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.06)"),
        yaxis=dict(gridcolor=GRID, zerolinecolor="rgba(255,255,255,0.08)", linecolor="rgba(255,255,255,0.06)"),
        legend=dict(bgcolor="rgba(10,14,26,0.6)", bordercolor="rgba(255,255,255,0.08)", borderwidth=1),
        margin=dict(t=24, b=24, l=8, r=8),
        hoverlabel=dict(bgcolor="#0d1b2a", bordercolor=G, font_color="#e2e8f0"),
    )
    base.update(kw)
    return base

# ─── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800;900&display=swap');

/* ── Base ── */
html, body, [data-testid="stAppViewContainer"], .main, .block-container {
    background: #060b14 !important;
    color: #e2e8f0 !important;
    font-family: 'Inter', 'Google Sans', -apple-system, sans-serif !important;
}
.stApp { background: linear-gradient(135deg,#060b14 0%,#0a1628 60%,#06101e 100%) !important; }
.block-container { padding-top: 1.5rem !important; max-width: 1400px !important; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg,#07101f 0%,#0a1525 100%) !important;
    border-right: 1px solid rgba(118,185,0,0.15) !important;
}
[data-testid="stSidebar"] * { color: #94a3b8 !important; }
[data-testid="stSidebar"] h1 { color: #e2e8f0 !important; font-size:17px !important; font-weight:700 !important; }
[data-testid="stSidebar"] h3 {
    color: #76b900 !important; font-size:10px !important; font-weight:700 !important;
    letter-spacing:2px !important; text-transform:uppercase !important; margin-top:22px !important;
}
[data-testid="stSidebar"] .stTextInput input,
[data-testid="stSidebar"] .stNumberInput input {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(255,255,255,0.09) !important;
    color: #e2e8f0 !important; border-radius:8px !important;
}
[data-testid="stSidebar"] hr { border-color: rgba(118,185,0,0.15) !important; }
[data-testid="stSidebar"] .stCaption { color: #3d4f63 !important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"] {
    background: rgba(10,20,38,0.7) !important;
    border-bottom: 1px solid rgba(255,255,255,0.06) !important;
    border-radius: 12px 12px 0 0 !important; gap:0 !important; padding:0 !important;
}
.stTabs [data-baseweb="tab"] {
    background: transparent !important; color: #4a5568 !important;
    font-size:13px !important; font-weight:500 !important;
    padding:14px 22px !important; border-bottom:3px solid transparent !important;
    transition: all 0.2s !important;
}
.stTabs [aria-selected="true"] {
    color: #76b900 !important; border-bottom:3px solid #76b900 !important;
    background: rgba(118,185,0,0.05) !important; font-weight:600 !important;
}
.stTabs [data-baseweb="tab"]:hover { color:#a3c840 !important; background:rgba(118,185,0,0.03) !important; }
.stTabs [data-baseweb="tab-panel"] { background:transparent !important; padding:28px 0 !important; }

/* ── Hero Header ── */
.hero {
    background: linear-gradient(135deg,#0a1628 0%,#0d2040 50%,#0a1525 100%);
    border:1px solid rgba(118,185,0,0.18); border-radius:16px;
    padding:28px 32px; margin-bottom:28px; position:relative; overflow:hidden;
    display:flex; align-items:center; gap:20px;
}
.hero::before {
    content:''; position:absolute; top:0; left:0; right:0; height:3px;
    background: linear-gradient(90deg,#76b900,#4285F4,#76b900);
    background-size:200%; animation: shimmer 4s linear infinite;
}
@keyframes shimmer { 0%{background-position:-200% 0} 100%{background-position:200% 0} }
.hero-glow {
    position:absolute; top:-60px; right:-60px; width:240px; height:240px;
    background:radial-gradient(circle,rgba(118,185,0,0.07) 0%,transparent 70%);
    pointer-events:none;
}
.hero-glow2 {
    position:absolute; bottom:-60px; left:30%; width:200px; height:200px;
    background:radial-gradient(circle,rgba(66,133,244,0.05) 0%,transparent 70%);
    pointer-events:none;
}
.hero-icon { font-size:52px; filter:drop-shadow(0 0 16px rgba(118,185,0,0.5)); flex-shrink:0; }
.hero-title {
    font-size:26px; font-weight:800; line-height:1.2;
    background:linear-gradient(90deg,#ffffff 0%,#a3c840 45%,#4285F4 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent; background-clip:text;
}
.hero-sub { font-size:13px; color:#4a6480; margin-top:5px; font-weight:400; }
.hero-badges { margin-top:12px; }
.hbadge {
    display:inline-block; padding:3px 10px; border-radius:20px;
    font-size:10px; font-weight:700; letter-spacing:0.5px; margin-right:6px;
}
.hb-gcp   { background:rgba(66,133,244,0.12); color:#4285F4; border:1px solid rgba(66,133,244,0.25); }
.hb-aws   { background:rgba(255,153,0,0.12);  color:#ff9900; border:1px solid rgba(255,153,0,0.25); }
.hb-azure { background:rgba(0,114,239,0.12);  color:#0072ef; border:1px solid rgba(0,114,239,0.25); }
.hb-nv    { background:rgba(118,185,0,0.12);  color:#76b900; border:1px solid rgba(118,185,0,0.25); }

/* ── KPI Cards ── */
.kpi {
    background:linear-gradient(135deg,rgba(10,22,40,0.95),rgba(13,27,55,0.9));
    border:1px solid rgba(255,255,255,0.06); border-radius:14px;
    padding:22px 20px; position:relative; overflow:hidden;
    margin-bottom:12px; transition:transform 0.2s,border-color 0.2s;
}
.kpi:hover { transform:translateY(-3px); }
.kpi::before { content:''; position:absolute; top:0; left:0; right:0; height:3px; border-radius:14px 14px 0 0; }
.kpi-g::before  { background:linear-gradient(90deg,#76b900,#a3c840); }
.kpi-g:hover    { border-color:rgba(118,185,0,0.3); }
.kpi-b::before  { background:linear-gradient(90deg,#4285F4,#6dd5ed); }
.kpi-b:hover    { border-color:rgba(66,133,244,0.3); }
.kpi-y::before  { background:linear-gradient(90deg,#fbbc04,#ffd200); }
.kpi-y:hover    { border-color:rgba(251,188,4,0.3); }
.kpi-p::before  { background:linear-gradient(90deg,#a78bfa,#c084fc); }
.kpi-p:hover    { border-color:rgba(167,139,250,0.3); }
.kpi-r::before  { background:linear-gradient(90deg,#ea4335,#ff6b6b); }
.kpi-r:hover    { border-color:rgba(234,67,53,0.3); }
.kpi-glow {
    position:absolute; bottom:-30px; right:-30px; width:100px; height:100px;
    border-radius:50%; opacity:0.35; pointer-events:none;
}
.kpi-g .kpi-glow { background:radial-gradient(#76b900,transparent); }
.kpi-b .kpi-glow { background:radial-gradient(#4285F4,transparent); }
.kpi-y .kpi-glow { background:radial-gradient(#fbbc04,transparent); }
.kpi-p .kpi-glow { background:radial-gradient(#a78bfa,transparent); }
.kpi-label { font-size:10px; font-weight:700; letter-spacing:1.5px; text-transform:uppercase; color:#3d5166; margin-bottom:8px; }
.kpi-icon  { font-size:20px; margin-bottom:8px; display:block; }
.kpi-val   { font-size:28px; font-weight:800; line-height:1; margin-bottom:6px; }
.kpi-g .kpi-val { color:#76b900; }
.kpi-b .kpi-val { color:#4285F4; }
.kpi-y .kpi-val { color:#ffd200; }
.kpi-p .kpi-val { color:#a78bfa; }
.kpi-r .kpi-val { color:#ff6b6b; }
.kpi-delta { font-size:11px; color:#3d5166; }

/* ── Section Headers ── */
.sec-hdr {
    display:flex; align-items:center; gap:10px;
    font-size:10px; font-weight:700; letter-spacing:2px; text-transform:uppercase;
    color:#4a6480; padding:20px 0 12px;
    border-bottom:1px solid rgba(255,255,255,0.05); margin-bottom:18px;
}
.sec-hdr::before {
    content:''; width:4px; height:16px;
    background:linear-gradient(180deg,#76b900,#4285F4);
    border-radius:2px; flex-shrink:0;
}

/* ── Recommendation Boxes ── */
.rec {
    background:rgba(8,16,30,0.7); border:1px solid rgba(255,255,255,0.05);
    border-left:3px solid #76b900; border-radius:10px;
    padding:14px 18px; margin:8px 0; transition:border-left-color 0.3s,background 0.2s;
}
.rec:hover { border-left-color:#4285F4; background:rgba(10,22,40,0.9); }
.rec-title { font-size:13px; font-weight:600; color:#c8d8e8; margin-bottom:5px; }
.rec-body  { font-size:12px; color:#4a6480; line-height:1.7; }

/* ── Streamlit native element overrides ── */
.stMetric {
    background:rgba(10,22,40,0.6) !important; border:1px solid rgba(255,255,255,0.06) !important;
    border-radius:10px !important; padding:16px !important;
}
.stMetric label { color:#4a5568 !important; font-size:11px !important; font-weight:600 !important; letter-spacing:1px !important; text-transform:uppercase !important; }
[data-testid="stMetricValue"] { color:#76b900 !important; font-weight:700 !important; }

.stTextInput input, .stNumberInput input {
    background:rgba(255,255,255,0.04) !important; border:1px solid rgba(255,255,255,0.09) !important;
    color:#e2e8f0 !important; border-radius:8px !important; font-family:inherit !important;
}
.stTextInput input:focus, .stNumberInput input:focus {
    border-color:#76b900 !important; box-shadow:0 0 0 2px rgba(118,185,0,0.12) !important;
}
[data-testid="stSlider"] > div > div > div > div { background:#76b900 !important; }
[data-testid="stSlider"] > div > div > div > div:last-child { background:#76b900 !important; }

.streamlit-expanderHeader {
    background:rgba(10,22,40,0.6) !important; border:1px solid rgba(255,255,255,0.06) !important;
    border-radius:8px !important; color:#64748b !important; font-size:13px !important;
}
.streamlit-expanderContent {
    background:rgba(6,11,20,0.8) !important; border:1px solid rgba(255,255,255,0.04) !important;
    border-top:none !important; border-radius:0 0 8px 8px !important;
}

div[data-baseweb="notification"] { background:rgba(66,133,244,0.08) !important; border:1px solid rgba(66,133,244,0.2) !important; border-radius:8px !important; }
div[data-baseweb="notification"] * { color:#93c5fd !important; }

hr { border-color:rgba(255,255,255,0.05) !important; }
label, p { color:#94a3b8 !important; }
h1,h2,h3,h4 { color:#e2e8f0 !important; }
caption, .stCaption { color:#3d5166 !important; }

[data-testid="stDataFrame"] { border:1px solid rgba(255,255,255,0.06) !important; border-radius:10px !important; overflow:hidden !important; }

.stDownloadButton > button {
    background:linear-gradient(90deg,#76b900,#4285F4) !important; color:#fff !important;
    border:none !important; border-radius:8px !important; font-weight:600 !important; padding:10px 24px !important;
    transition:opacity 0.2s !important;
}
.stDownloadButton > button:hover { opacity:0.85 !important; }

::-webkit-scrollbar { width:5px; height:5px; }
::-webkit-scrollbar-track { background:rgba(255,255,255,0.02); }
::-webkit-scrollbar-thumb { background:rgba(118,185,0,0.3); border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:rgba(118,185,0,0.5); }

/* sidebar selectbox dropdown */
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background:rgba(255,255,255,0.04) !important; border:1px solid rgba(255,255,255,0.09) !important;
    border-radius:8px !important;
}
</style>
""", unsafe_allow_html=True)


# ─── Helpers ─────────────────────────────────────────────────────────────────

def fmt(n):
    if abs(n) >= 1_000_000:
        return f"${n/1_000_000:.2f}M"
    if abs(n) >= 1_000:
        return f"${n/1_000:.1f}K"
    return f"${n:.0f}"

def kpi(icon, label, value, delta, css="kpi-g"):
    st.markdown(f"""
    <div class="kpi {css}">
        <span class="kpi-icon">{icon}</span>
        <div class="kpi-label">{label}</div>
        <div class="kpi-val">{value}</div>
        <div class="kpi-delta">{delta}</div>
        <div class="kpi-glow"></div>
    </div>""", unsafe_allow_html=True)

def sec(label):
    st.markdown(f'<div class="sec-hdr">{label}</div>', unsafe_allow_html=True)

def rec(icon, title, body):
    st.markdown(f"""
    <div class="rec">
        <div class="rec-title">{icon} {title}</div>
        <div class="rec-body">{body}</div>
    </div>""", unsafe_allow_html=True)


# ─── Sidebar ─────────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("""
    <div style="display:flex;align-items:center;gap:12px;padding:4px 0 8px">
        <span style="font-size:32px;filter:drop-shadow(0 0 10px rgba(118,185,0,0.6))">🤖</span>
        <div>
            <div style="font-size:15px;font-weight:800;color:#e2e8f0">GenAI TCO</div>
            <div style="font-size:10px;color:#3d5166;letter-spacing:1px;text-transform:uppercase">Enterprise Calculator</div>
        </div>
    </div>""", unsafe_allow_html=True)
    st.divider()

    st.markdown("### Company Profile")
    company_name = st.text_input("Company Name", "Acme Corp")
    industry = st.selectbox("Industry", [
        "Financial Services", "Healthcare", "Retail / E-commerce",
        "Technology", "Manufacturing", "Telecom", "Insurance", "Other"])
    num_employees = st.number_input("Total Employees", 100, 500_000, 5000, step=500)

    st.markdown("### Use Case")
    use_case = st.selectbox("Primary Use Case", [
        "Customer Support Automation",
        "Knowledge Base / Internal Q&A",
        "Document Processing & Summarization",
        "Code Generation / Developer Tooling",
        "Sales & Marketing Personalization",
        "Fraud Detection & Risk Analysis",
    ])

    st.markdown("### Scale")
    monthly_queries = st.number_input("Monthly AI Queries", 1000, 50_000_000, 500_000, step=10_000)
    avg_input_tokens = st.slider("Avg Input Tokens / Query", 100, 8000, 800, step=100)
    avg_output_tokens = st.slider("Avg Output Tokens / Query", 50, 4000, 300, step=50)

    st.markdown("### Deployment")
    cloud_provider = st.selectbox("Cloud Provider", ["GCP", "AWS", "Azure"])
    deployment_type = st.selectbox("Deployment Type", [
        "Cloud API (Managed)", "Cloud VM (Self-managed)", "Self-hosted (On-Prem)"])
    model_approach = st.selectbox("Model Approach", [
        "API / Third-party LLM", "Fine-tuned Open Source", "Self-hosted OSS"])

    st.divider()
    st.markdown("""
    <div style="text-align:center;padding:8px 0">
        <span style="background:rgba(118,185,0,0.1);color:#76b900;border:1px solid rgba(118,185,0,0.2);
               padding:2px 8px;border-radius:12px;font-size:10px;font-weight:700;">v1.0</span>
        <span style="background:rgba(66,133,244,0.1);color:#4285F4;border:1px solid rgba(66,133,244,0.2);
               padding:2px 8px;border-radius:12px;font-size:10px;font-weight:700;margin-left:6px;">Multi-Cloud</span>
    </div>""", unsafe_allow_html=True)


# ─── Tabs ────────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊  Executive Dashboard",
    "💰  Cost Components",
    "🔄  Scenario Comparison",
    "🖥️  GPU Optimization",
    "📈  ROI & Payback",
    "📋  Docs & Export",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Cost Components  (computed first — feeds all other tabs)
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    sec("Infrastructure")
    c1, c2, c3 = st.columns(3)
    with c1:
        gpu_type = st.selectbox("GPU Type", list(GPU_PRICING.keys()), key="gpu_type")
        gpu_count = st.number_input("GPU Count", 1, 256, 4, key="gpu_count")
    with c2:
        utilization_pct = st.slider("GPU Utilization %", 10, 100, 60, key="util")
        pricing_tier = st.selectbox("Pricing Tier", ["on_demand", "reserved_1yr", "reserved_3yr"], key="tier",
            format_func=lambda x: {"on_demand": "On-Demand", "reserved_1yr": "Reserved 1yr", "reserved_3yr": "Reserved 3yr"}[x])
    with c3:
        storage_gb = st.number_input("Storage (GB)", 100, 500_000, 5000, step=100, key="storage_gb")
        networking_egress_gb = st.number_input("Monthly Egress (GB)", 10, 100_000, 500, key="egress_gb")

    infra = compute_infrastructure_cost(
        deployment_type, gpu_type, gpu_count, utilization_pct,
        pricing_tier, storage_gb, cloud_provider, networking_egress_gb)

    sec("Model Cost")
    ca, cb = st.columns(2)
    with ca:
        if model_approach == "API / Third-party LLM":
            selected_model = st.selectbox("LLM Model", list(LLM_API_PRICING.keys()), key="api_model")
            model_api_cost = compute_api_model_cost(selected_model, monthly_queries, avg_input_tokens, avg_output_tokens)
            st.info(f"Token cost: **{fmt(model_api_cost)}/month** at {monthly_queries:,} queries")
        elif model_approach == "Fine-tuned Open Source":
            ft_model_size = st.selectbox("Base Model Size", list(FINE_TUNING_GPU_HOURS_PER_BILLION_TOKENS.keys()), key="ft_size")
            training_tokens_b = st.slider("Training Data (Billion Tokens)", 0.1, 10.0, 1.0, step=0.1, key="train_tok")
            ft_one_time = compute_fine_tuning_cost(ft_model_size, training_tokens_b, gpu_type, pricing_tier)
            inference_gpu_cost = compute_gpu_monthly_cost(gpu_type, max(1, gpu_count // 2), utilization_pct, pricing_tier)
            model_api_cost = inference_gpu_cost
            st.info(f"One-time fine-tuning: **{fmt(ft_one_time)}** · Inference: **{fmt(model_api_cost)}/month**")
        else:
            oss_model = st.selectbox("OSS Model", list(OPEN_SOURCE_MODELS.keys()), key="oss_model")
            model_api_cost = compute_gpu_monthly_cost(gpu_type, gpu_count, utilization_pct, pricing_tier)
            m = OPEN_SOURCE_MODELS[oss_model]
            st.info(f"~{m['tokens_per_sec_a100']} tokens/sec on A100 · {m['gpu_memory_gb']} GB VRAM required")
    with cb:
        if model_approach == "API / Third-party LLM":
            p = LLM_API_PRICING.get(selected_model, {})
            rows = [
                ("Provider", p.get("provider", "—")),
                ("Input price", f"${p.get('input_per_1m', 0):.2f} / 1M tokens"),
                ("Output price", f"${p.get('output_per_1m', 0):.2f} / 1M tokens"),
                ("Context window", f"{p.get('context_window_k', '?')}K tokens"),
                ("Quality score", f"{p.get('quality_score', '?')} / 10"),
            ]
            st.table(pd.DataFrame(rows, columns=["Attribute", "Value"]).set_index("Attribute"))

    sec("Data Pipeline")
    d1, d2, d3 = st.columns(3)
    with d1:
        monthly_data_ingestion_gb = st.number_input("Monthly Data Ingestion (GB)", 1, 100_000, 200, key="ingest_gb")
        transformation_hours = st.number_input("Monthly ETL Compute Hours", 1, 10_000, 100, key="etl_h")
    with d2:
        vector_db_vectors_m = st.number_input("Vector DB Size (Million Vectors)", 0.1, 1000.0, 5.0, step=0.5, key="vdb_m")
        monthly_vector_queries_m = st.number_input("Monthly Vector Queries (M)", 0.1, 1000.0, 2.0, step=0.1, key="vq_m")
    with d3:
        vector_db_type = st.selectbox("Vector DB", list(VECTOR_DB_PRICING.keys()), key="vdb_type")

    pipeline = compute_data_pipeline_cost(
        monthly_data_ingestion_gb, transformation_hours,
        vector_db_vectors_m, monthly_vector_queries_m, vector_db_type, cloud_provider)

    sec("Engineering & Labor")
    e1, e2 = st.columns(2)
    with e1:
        team_comp = {}
        for role in LABOR_RATES:
            team_comp[role] = st.number_input(role, 0, 20, 1 if role in ("ML Engineer", "Data Engineer") else 0, key=f"hc_{role}")
    with e2:
        labor_alloc_pct = st.slider("% Time Allocated to GenAI", 10, 100, 70, key="labor_alloc")
        st.caption("Reduce if team works across multiple projects")

    labor = compute_labor_cost(team_comp, labor_alloc_pct)

    sec("Maintenance & Operations")
    maintenance_pct = st.slider("Maintenance as % of Infra + Pipeline Cost", 5, 30, 15, key="maint_pct")
    maintenance = compute_maintenance_cost(infra["total"], pipeline["total"], maintenance_pct)

    tco = compute_total_tco(infra, model_api_cost, pipeline, labor, maintenance)

    sec("Monthly Cost Breakdown")
    breakdown = tco["breakdown"]
    fig_wf = go.Figure(go.Waterfall(
        orientation="v",
        measure=["relative"] * len(breakdown) + ["total"],
        x=list(breakdown.keys()) + ["Total TCO"],
        y=list(breakdown.values()) + [tco["monthly"]],
        text=[fmt(v) for v in list(breakdown.values()) + [tco["monthly"]]],
        textposition="outside",
        textfont=dict(color="#e2e8f0", size=11),
        connector={"line": {"color": "rgba(255,255,255,0.1)", "width": 1}},
        increasing={"marker": {"color": G, "line": {"color": G, "width": 1}}},
        totals={"marker": {"color": B, "line": {"color": B, "width": 1}}},
    ))
    fig_wf.update_layout(**chart_layout(height=400, yaxis_title="USD / Month", showlegend=False))
    st.plotly_chart(fig_wf, use_container_width=True)

    col_a, col_b, col_c = st.columns(3)
    with col_a:
        with st.expander("🖥️ Infrastructure Detail"):
            st.table(pd.DataFrame([
                ("GPU Compute", fmt(infra["gpu_compute"])),
                ("Storage", fmt(infra["storage"])),
                ("Networking", fmt(infra["networking"])),
                ("Cloud Overhead", fmt(infra["cloud_overhead"])),
            ], columns=["Component", "Monthly"]).set_index("Component"))
    with col_b:
        with st.expander("🔄 Data Pipeline Detail"):
            st.table(pd.DataFrame([
                ("Ingestion", fmt(pipeline["ingestion"])),
                ("ETL / Transform", fmt(pipeline["transformation"])),
                ("Vector DB Storage", fmt(pipeline["vector_db_storage"])),
                ("Vector DB Queries", fmt(pipeline["vector_db_queries"])),
                ("Raw Storage", fmt(pipeline["raw_storage"])),
            ], columns=["Component", "Monthly"]).set_index("Component"))
    with col_c:
        with st.expander("👥 Labor Detail"):
            rows = [(role, fmt(cost)) for role, cost in labor["breakdown"].items() if cost > 0]
            if rows:
                st.table(pd.DataFrame(rows, columns=["Role", "Monthly"]).set_index("Role"))


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Executive Dashboard
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    # Hero
    st.markdown(f"""
    <div class="hero">
        <div class="hero-glow"></div><div class="hero-glow2"></div>
        <div class="hero-icon">🤖</div>
        <div>
            <div class="hero-title">{company_name} · GenAI TCO & ROI</div>
            <div class="hero-sub">{use_case} &nbsp;·&nbsp; {cloud_provider} &nbsp;·&nbsp; {monthly_queries:,} queries / month</div>
            <div class="hero-badges">
                <span class="hbadge hb-gcp">◆ Google Cloud</span>
                <span class="hbadge hb-aws">◆ AWS</span>
                <span class="hbadge hb-azure">◆ Azure</span>
                <span class="hbadge hb-nv">▲ NVIDIA GPU</span>
            </div>
        </div>
    </div>""", unsafe_allow_html=True)

    # KPI row
    k1, k2, k3, k4 = st.columns(4)
    cost_per_query = tco["monthly"] / monthly_queries if monthly_queries else 0
    with k1: kpi("💰", "Monthly TCO",  fmt(tco["monthly"]),  f"Annual: {fmt(tco['annual'])}", "kpi-g")
    with k2: kpi("🔍", "Cost Per Query", f"${cost_per_query:.5f}", f"{monthly_queries:,} queries/mo", "kpi-b")
    with k3: kpi("📅", "Annual TCO", fmt(tco["annual"]), f"3-Year: {fmt(tco['three_year'])}", "kpi-y")
    with k4: kpi("🚀", "Model Approach", model_approach.split(" ")[0], deployment_type, "kpi-p")

    st.markdown("<br>", unsafe_allow_html=True)
    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown("#### Cost Distribution")
        fig_pie = go.Figure(go.Pie(
            labels=list(tco["breakdown"].keys()),
            values=list(tco["breakdown"].values()),
            hole=0.52,
            marker=dict(colors=COLORS, line=dict(color="#060b14", width=2)),
            textfont=dict(color="#e2e8f0", size=11),
            hovertemplate="<b>%{label}</b><br>%{value:$,.0f}<br>%{percent}<extra></extra>",
        ))
        fig_pie.add_annotation(text=f"<b>{fmt(tco['monthly'])}</b><br><span style='font-size:10px'>/ month</span>",
                               x=0.5, y=0.5, showarrow=False, font=dict(size=16, color="#e2e8f0"), align="center")
        fig_pie.update_layout(**chart_layout(height=360, showlegend=True,
                                              legend=dict(orientation="v", x=1.02, y=0.5)))
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_r:
        st.markdown("#### 3-Year Cumulative TCO")
        proj = project_costs_over_time(tco["monthly"], 36)
        df_proj = pd.DataFrame(proj)
        fig_tco = go.Figure()
        fig_tco.add_trace(go.Bar(
            x=df_proj["month"], y=df_proj["monthly"], name="Monthly",
            marker_color=f"rgba(118,185,0,0.35)", yaxis="y2"))
        fig_tco.add_trace(go.Scatter(
            x=df_proj["month"], y=df_proj["cumulative"], name="Cumulative",
            line=dict(color=G, width=2.5),
            fill="tozeroy", fillcolor="rgba(118,185,0,0.08)"))
        layout = chart_layout(height=360)
        layout["yaxis"] = dict(**layout.get("yaxis", {}), title="Cumulative ($)", color=FONT)
        layout["yaxis2"] = dict(title="Monthly ($)", overlaying="y", side="right",
                                 gridcolor=GRID, color=FONT)
        layout["legend"] = dict(orientation="h", y=-0.18, bgcolor="rgba(0,0,0,0)")
        fig_tco.update_layout(**layout)
        st.plotly_chart(fig_tco, use_container_width=True)

    # Summary table
    st.markdown("#### Cost Summary")
    summary = pd.DataFrame({
        "Period":    ["Monthly", "Quarterly", "Annual", "3-Year", "5-Year"],
        "TCO":       [fmt(tco["monthly"]), fmt(tco["monthly"]*3), fmt(tco["annual"]), fmt(tco["three_year"]), fmt(tco["five_year"])],
        "$/Query":   [f"${tco['monthly']/monthly_queries:.5f}"]*5 if monthly_queries else ["—"]*5,
        "Top Driver":[max(tco["breakdown"], key=tco["breakdown"].get)]*5,
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

    # Recommendations
    st.markdown("#### Strategic Recommendations")
    largest = max(tco["breakdown"], key=tco["breakdown"].get)
    if largest == "Labor":
        rec("🧑‍💼", "Labor Dominates Costs", "Consider increasing automation and evaluating managed MLOps services to reduce engineering overhead.")
    if largest == "Model / API":
        rec("💡", "Optimize Model Selection", "API spend is your top cost. Route simple queries to Claude Haiku or Gemini Flash and reserve premium models for complex tasks — typical savings: 40–60%.")
    if infra["gpu_compute"] > 0 and utilization_pct < 40:
        rec("🖥️", f"Low GPU Utilization ({utilization_pct}%)", "Switch batch workloads to spot/preemptible instances and scale down idle capacity. Target 70%+ utilization for cost efficiency.")
    if deployment_type == "Cloud API (Managed)":
        rec("☁️", "Evaluate Reserved Pricing", "Committing to 1-year reserved GPU instances typically saves 35–45% vs on-demand rates at your usage scale.")
    rec("📊", "Monitor & Iterate", "Set monthly TCO budget alerts. A 10% reduction in token usage at 500K queries/month yields measurable annual savings.")


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Scenario Comparison
# ════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("""<div class="hero" style="padding:20px 28px;margin-bottom:24px">
        <div class="hero-glow"></div>
        <div style="font-size:28px">🔄</div>
        <div>
            <div style="font-size:18px;font-weight:700;color:#e2e8f0">Build vs Buy &nbsp;·&nbsp; Cloud vs On-Prem &nbsp;·&nbsp; Model Alternatives</div>
            <div style="font-size:12px;color:#4a6480;margin-top:4px">Side-by-side cost analysis across all deployment strategies</div>
        </div>
    </div>""", unsafe_allow_html=True)

    base_inputs = dict(monthly_queries=monthly_queries, avg_input_tokens=avg_input_tokens,
                       avg_output_tokens=avg_output_tokens, storage_gb=storage_gb,
                       networking_egress_gb=networking_egress_gb, utilization_pct=utilization_pct)
    scenarios = compute_scenario_comparison(base_inputs)

    fig_sc = go.Figure()
    cats = ["Model / Infra Cost", "Infrastructure Cost", "Total Monthly"]
    for i, (name, vals) in enumerate(scenarios.items()):
        fig_sc.add_trace(go.Bar(
            name=name,
            x=cats,
            y=[vals["model"], vals["infra"], vals["total"]],
            marker=dict(color=COLORS[i], opacity=0.85),
            text=[fmt(vals["model"]), fmt(vals["infra"]), fmt(vals["total"])],
            textposition="outside", textfont=dict(color="#e2e8f0", size=10),
        ))
    fig_sc.update_layout(**chart_layout(barmode="group", height=420, yaxis_title="USD / Month",
                                         legend=dict(orientation="h", y=-0.2)))
    st.plotly_chart(fig_sc, use_container_width=True)

    lowest = min(scenarios, key=lambda k: scenarios[k]["total"])
    rows = []
    for name, vals in scenarios.items():
        diff = tco["monthly"] - vals["total"]
        rows.append({"Scenario": name, "Model Cost": fmt(vals["model"]), "Infra Cost": fmt(vals["infra"]),
                      "Total/Month": fmt(vals["total"]), "Annual": fmt(vals["total"]*12),
                      "vs Current": ("+"+fmt(diff) if diff > 0 else fmt(diff)), "":("✅ Best" if name==lowest else "")})
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    sec("Model-by-Model API Cost Comparison")
    st.caption(f"{monthly_queries:,} queries/month · {avg_input_tokens} input + {avg_output_tokens} output tokens")

    model_rows = []
    for mname, mp in LLM_API_PRICING.items():
        cost = compute_api_model_cost(mname, monthly_queries, avg_input_tokens, avg_output_tokens)
        model_rows.append({"Model": mname, "Provider": mp["provider"],
                           "Monthly": fmt(cost), "Annual": fmt(cost*12),
                           "Input $/1M": f"${mp['input_per_1m']}", "Output $/1M": f"${mp['output_per_1m']}",
                           "Context": f"{mp['context_window_k']}K", "Quality": "⭐"*mp["quality_score"], "_s": cost})
    df_models = pd.DataFrame(model_rows).sort_values("_s").drop(columns=["_s"])
    st.dataframe(df_models, use_container_width=True, hide_index=True)

    st.divider()
    sec("3-Year Cloud vs On-Prem Projection")
    cloud_m  = list(scenarios.values())[0]["total"] + labor["total"] + pipeline["total"]
    onprem_m = list(scenarios.values())[2]["total"] + labor["total"] + pipeline["total"]
    capex    = GPU_PRICING.get(gpu_type, {}).get("on_demand_hourly", 3.67) * gpu_count * 8760 * 0.2

    months_r   = list(range(1, 37))
    cloud_cum  = [cloud_m * m for m in months_r]
    onprem_cum = [capex + onprem_m * m for m in months_r]
    fig_cvo = go.Figure()
    fig_cvo.add_trace(go.Scatter(x=months_r, y=cloud_cum, name="Cloud API",
                                  line=dict(color=B, width=2.5),
                                  fill="tozeroy", fillcolor="rgba(66,133,244,0.06)"))
    fig_cvo.add_trace(go.Scatter(x=months_r, y=onprem_cum, name="On-Prem (incl. CapEx)",
                                  line=dict(color=R, width=2.5),
                                  fill="tozeroy", fillcolor="rgba(234,67,53,0.05)"))
    cross = next((m for m,(c,o) in enumerate(zip(cloud_cum, onprem_cum),1) if c > o), None)
    if cross:
        fig_cvo.add_vline(x=cross, line_dash="dash", line_color=G,
                          annotation_text=f"Crossover ▲ Month {cross}",
                          annotation_font=dict(color=G, size=11), annotation_position="top right")
    fig_cvo.update_layout(**chart_layout(height=380, yaxis_title="Cumulative Cost ($)", xaxis_title="Month"))
    st.plotly_chart(fig_cvo, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — GPU Optimization
# ════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("""<div class="hero" style="padding:20px 28px;margin-bottom:24px">
        <div class="hero-glow"></div>
        <div style="font-size:28px">🖥️</div>
        <div>
            <div style="font-size:18px;font-weight:700;color:#e2e8f0">NVIDIA GPU Cost Optimization</div>
            <div style="font-size:12px;color:#4a6480;margin-top:4px">Utilization curves · Reserved vs on-demand · Idle waste analysis</div>
        </div>
    </div>""", unsafe_allow_html=True)

    g1, g2 = st.columns(2)
    with g1:
        gpu_opt_type  = st.selectbox("Select GPU", list(GPU_PRICING.keys()), key="gpu_opt")
        gpu_opt_count = st.slider("GPU Count", 1, 32, 4, key="gpu_opt_count")
    with g2:
        util_range = st.slider("Utilization Range (%)", 10, 100, (20, 90), key="util_range")

    gi = GPU_PRICING[gpu_opt_type]

    # Spec cards
    sp1, sp2, sp3, sp4 = st.columns(4)
    with sp1: kpi("🧠", "GPU Memory",       f"{gi['memory_gb']} GB",             f"{gpu_opt_type}", "kpi-g")
    with sp2: kpi("⚡", "BF16 Performance", f"{gi['flops_bf16_tflops']} TFLOPS", "AI Compute Power", "kpi-b")
    with sp3: kpi("💵", "On-Demand Rate",   f"${gi['on_demand_hourly']:.2f}/hr",  "Per GPU", "kpi-y")
    with sp4: kpi("📅", "Reserved 1yr",     f"${gi['reserved_1yr_hourly']:.2f}/hr", f"Save {round((1-gi['reserved_1yr_hourly']/gi['on_demand_hourly'])*100)}%", "kpi-p")

    # Utilization vs cost
    util_pts = list(range(10, 101, 5))
    data = []
    for tier, label in [("on_demand","On-Demand"),("reserved_1yr","Reserved 1yr"),("reserved_3yr","Reserved 3yr")]:
        for u in util_pts:
            data.append({"Utilization (%)": u,
                          "Monthly Cost ($)": compute_gpu_monthly_cost(gpu_opt_type, gpu_opt_count, u, tier),
                          "Tier": label})
    df_util = pd.DataFrame(data)
    fig_util = px.line(df_util, x="Utilization (%)", y="Monthly Cost ($)", color="Tier",
                       color_discrete_sequence=[R, G, B], markers=True,
                       hover_data={"Monthly Cost ($)": ":$,.0f"})
    fig_util.add_vrect(x0=util_range[0], x1=util_range[1],
                       fillcolor="rgba(118,185,0,0.05)", line_width=0,
                       annotation_text="Your range", annotation_position="top left",
                       annotation_font=dict(color=G, size=10))
    fig_util.update_layout(**chart_layout(height=380, yaxis_title="Monthly Cost ($)"))
    st.plotly_chart(fig_util, use_container_width=True)

    # GPU comparison table
    sec("All GPU Types — Side-by-Side")
    gpu_rows = []
    for gname, gdata in GPU_PRICING.items():
        od = compute_gpu_monthly_cost(gname, gpu_opt_count, utilization_pct, "on_demand")
        r1 = compute_gpu_monthly_cost(gname, gpu_opt_count, utilization_pct, "reserved_1yr")
        r3 = compute_gpu_monthly_cost(gname, gpu_opt_count, utilization_pct, "reserved_3yr")
        gpu_rows.append({"GPU": gname, "Memory": f"{gdata['memory_gb']} GB",
                          "BF16 TFLOPS": gdata["flops_bf16_tflops"],
                          "On-Demand/mo": fmt(od), "Reserved 1yr/mo": fmt(r1),
                          "Reserved 3yr/mo": fmt(r3),
                          "1yr Savings": f"{round((1-r1/od)*100) if od else 0}%",
                          "TFLOPS/$": f"{gdata['flops_bf16_tflops']/gdata['on_demand_hourly']:.1f}"})
    st.dataframe(pd.DataFrame(gpu_rows), use_container_width=True, hide_index=True)

    # Idle waste
    sec("Impact of Idle GPU Time")
    idle_data = [{"Utilization (%)": u,
                   "Active Cost ($)": compute_gpu_monthly_cost(gpu_opt_type, gpu_opt_count, u, "on_demand"),
                   "Wasted ($)": compute_gpu_monthly_cost(gpu_opt_type, gpu_opt_count, u, "on_demand") * (1-u/100)*0.5}
                 for u in range(10, 101, 10)]
    df_idle = pd.DataFrame(idle_data)
    fig_idle = go.Figure()
    fig_idle.add_trace(go.Bar(x=df_idle["Utilization (%)"], y=df_idle["Active Cost ($)"],
                               name="Active Cost", marker_color=G, opacity=0.8))
    fig_idle.add_trace(go.Bar(x=df_idle["Utilization (%)"], y=df_idle["Wasted ($)"],
                               name="Idle Waste", marker_color=R, opacity=0.7))
    fig_idle.update_layout(**chart_layout(barmode="overlay", height=320, yaxis_title="Monthly Cost ($)"))
    st.plotly_chart(fig_idle, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — ROI & Payback
# ════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("""<div class="hero" style="padding:20px 28px;margin-bottom:24px">
        <div class="hero-glow"></div>
        <div style="font-size:28px">📈</div>
        <div>
            <div style="font-size:18px;font-weight:700;color:#e2e8f0">ROI & Business Case Analysis</div>
            <div style="font-size:12px;color:#4a6480;margin-top:4px">Payback period · NPV · IRR · 5-year value projection</div>
        </div>
    </div>""", unsafe_allow_html=True)

    sec("Legacy System Baseline")
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        legacy_type  = st.selectbox("Legacy System Type", list(LEGACY_SYSTEM_BENCHMARKS.keys()), key="legacy_type")
        legacy_units = st.number_input("Units (agents / seats / licenses)", 1, 10_000, 50, key="legacy_units")
        legacy_monthly = LEGACY_SYSTEM_BENCHMARKS[legacy_type] * legacy_units / 12
        st.metric("Estimated Legacy Monthly Cost", fmt(legacy_monthly))
    with r1c2:
        initial_investment = st.number_input("Initial Setup Investment ($)", 0, 5_000_000, 50_000, step=5000, key="initial_inv")
        st.caption("Integration · data prep · training · infra setup · change management")

    sec("Productivity Gains")
    p1, p2, p3, p4 = st.columns(4)
    with p1: employees_affected = st.number_input("Employees Affected", 1, 10_000, 100, key="emp_aff")
    with p2: hours_saved_monthly = st.slider("Hours Saved / Employee / Month", 0, 80, 15, key="hrs_saved")
    with p3: avg_hourly_rate = st.number_input("Avg Hourly Rate ($)", 10, 200, 45, key="hrly_rate")
    with p4: automation_rate_pct = st.slider("Automation Rate %", 0, 100, 40, key="auto_rate")
    prod_gains = compute_productivity_gains(employees_affected, hours_saved_monthly, avg_hourly_rate, automation_rate_pct)

    sec("Cost Avoidance")
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

    sec("Revenue Impact")
    rv1, rv2, rv3 = st.columns(3)
    with rv1:
        monthly_interactions  = st.number_input("Monthly Customer Interactions", 100, 10_000_000, 50_000, step=1000, key="cust_int")
        avg_rev_per_customer  = st.number_input("Avg Revenue Per Customer ($)", 1, 10_000, 120, key="avg_rev")
    with rv2:
        baseline_csat = st.slider("Baseline CSAT Score", 1, 10, 6, key="base_csat")
        target_csat   = st.slider("Target CSAT Score",   1, 10, 8, key="tgt_csat")
    with rv3:
        churn_reduction_pct = st.slider("Churn Reduction %", 0, 50, 10, key="churn_red")
        upsell_lift_pct     = st.slider("Upsell Lift %",     0, 20,  5, key="upsell_lift")
    rev_impact = compute_revenue_impact(monthly_interactions, baseline_csat, target_csat,
                                        avg_rev_per_customer, churn_reduction_pct, upsell_lift_pct)

    roi_result = compute_full_roi(tco["monthly"], prod_gains, cost_avoid, rev_impact, initial_investment, months=60)
    payback    = roi_result["payback_months"]
    payback_str = f"{payback:.1f} mo" if payback != float("inf") else "Never"

    st.markdown("<br>", unsafe_allow_html=True)
    kp1, kp2, kp3, kp4 = st.columns(4)
    with kp1: kpi("⏱️", "Payback Period",    payback_str,                                       "Months to break even",          "kpi-b")
    with kp2: kpi("💚", "Monthly Net Benefit", fmt(roi_result["monthly_net"]),                  "Benefits − TCO",                 "kpi-g" if roi_result["monthly_net"]>0 else "kpi-r")
    with kp3: kpi("📐", "3-Year NPV",         fmt(abs(roi_result["npv"]))+(" ✅" if roi_result["npv"]>0 else " ⚠️"), "10% discount rate", "kpi-g" if roi_result["npv"]>0 else "kpi-r")
    with kp4: kpi("📊", "IRR",                f"{roi_result['irr_pct']:.1f}%" if roi_result["irr_pct"] else "—", "Internal Rate of Return", "kpi-y")

    col_ben, col_time = st.columns(2)
    with col_ben:
        st.markdown("#### Monthly Benefits Breakdown")
        fig_ben = go.Figure(go.Pie(
            labels=list(roi_result["benefit_breakdown"].keys()),
            values=list(roi_result["benefit_breakdown"].values()),
            hole=0.5,
            marker=dict(colors=[G, B, P], line=dict(color="#060b14", width=2)),
            textfont=dict(color="#e2e8f0", size=11),
            hovertemplate="<b>%{label}</b><br>$%{value:,.0f}<br>%{percent}<extra></extra>",
        ))
        total_ben = sum(roi_result["benefit_breakdown"].values())
        fig_ben.add_annotation(text=f"<b>{fmt(total_ben)}</b><br><span style='font-size:9px'>/ month</span>",
                               x=0.5, y=0.5, showarrow=False, font=dict(size=14, color="#e2e8f0"))
        fig_ben.update_layout(**chart_layout(height=350, showlegend=True,
                                              legend=dict(orientation="h", y=-0.12)))
        st.plotly_chart(fig_ben, use_container_width=True)

    with col_time:
        st.markdown("#### 5-Year Cumulative Value")
        df_tl = pd.DataFrame(roi_result["timeline"])
        fig_roi = go.Figure()
        fig_roi.add_trace(go.Scatter(x=df_tl["month"], y=df_tl["cumulative_benefit"],
                                      name="Cumulative Benefit", line=dict(color=G, width=2.5),
                                      fill="tozeroy", fillcolor="rgba(118,185,0,0.08)"))
        fig_roi.add_trace(go.Scatter(x=df_tl["month"], y=df_tl["cumulative_cost"],
                                      name="Cumulative Cost", line=dict(color=R, width=2.5),
                                      fill="tozeroy", fillcolor="rgba(234,67,53,0.06)"))
        if payback != float("inf") and payback <= 60:
            fig_roi.add_vline(x=payback, line_dash="dash", line_color=Y,
                              annotation_text=f"Payback ▲ M{payback:.0f}",
                              annotation_font=dict(color=Y, size=10), annotation_position="top left")
        fig_roi.update_layout(**chart_layout(height=350, yaxis_title="Cumulative Value ($)",
                                              xaxis_title="Month", legend=dict(orientation="h", y=-0.18)))
        st.plotly_chart(fig_roi, use_container_width=True)

    st.markdown("#### ROI % Over Time")
    fig_rp = go.Figure()
    fig_rp.add_trace(go.Scatter(x=df_tl["month"], y=df_tl["roi_pct"],
                                 line=dict(color=P, width=2.5),
                                 fill="tozeroy", fillcolor="rgba(167,139,250,0.08)",
                                 hovertemplate="Month %{x}<br>ROI: %{y:.1f}%<extra></extra>"))
    fig_rp.add_hline(y=0,   line_dash="dash", line_color="rgba(255,255,255,0.15)")
    fig_rp.add_hline(y=100, line_dash="dot",  line_color=G,
                     annotation_text="100% ROI", annotation_font=dict(color=G, size=10),
                     annotation_position="top right")
    fig_rp.update_layout(**chart_layout(height=280, yaxis_title="ROI (%)", xaxis_title="Month"))
    st.plotly_chart(fig_rp, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — Docs & Export
# ════════════════════════════════════════════════════════════════════════════

with tab6:
    st.markdown("""<div class="hero" style="padding:20px 28px;margin-bottom:24px">
        <div class="hero-glow"></div>
        <div style="font-size:28px">📋</div>
        <div>
            <div style="font-size:18px;font-weight:700;color:#e2e8f0">Assumptions, Formulas & Export</div>
            <div style="font-size:12px;color:#4a6480;margin-top:4px">Pricing reference · Cost formulas · Architecture notes · CSV export</div>
        </div>
    </div>""", unsafe_allow_html=True)

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
2. **Token compression** — strip whitespace, use concise system prompts
3. **Model routing** — lightweight model for simple queries, premium for complex
4. **Batch processing** — group non-realtime requests for cheaper batch APIs
5. **Reserved capacity** — 1yr commit = 35–45% savings vs on-demand
        """)

    with st.expander("📊 Export Current Scenario"):
        export = {
            "Company": company_name, "Industry": industry, "Use Case": use_case,
            "Monthly Queries": monthly_queries, "Cloud Provider": cloud_provider,
            "Deployment": deployment_type, "Model Approach": model_approach,
            "Monthly TCO ($)": round(tco["monthly"], 2), "Annual TCO ($)": round(tco["annual"], 2),
            "3-Year TCO ($)": round(tco["three_year"], 2),
            "Infrastructure ($)": round(infra["total"], 2), "Model/API ($)": round(model_api_cost, 2),
            "Data Pipeline ($)": round(pipeline["total"], 2), "Labor ($)": round(labor["total"], 2),
            "Maintenance ($)": round(maintenance, 2),
            "Monthly Benefits ($)": round(roi_result["monthly_benefits"], 2),
            "Payback (months)": round(roi_result["payback_months"], 1) if roi_result["payback_months"] != float("inf") else "N/A",
            "NPV ($)": round(roi_result["npv"], 2),
        }
        df_exp = pd.DataFrame([export]).T.reset_index()
        df_exp.columns = ["Metric", "Value"]
        st.dataframe(df_exp, use_container_width=True, hide_index=True)
        st.download_button(
            "⬇️  Download as CSV", df_exp.to_csv(index=False),
            f"genai_tco_{company_name.replace(' ','_')}.csv", "text/csv")
