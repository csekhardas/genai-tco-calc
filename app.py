import streamlit as st
import pandas as pd
import numpy as np
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

# ─── Styling ────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 12px; padding: 20px; color: white;
        text-align: center; margin-bottom: 16px;
    }
    .metric-card .label { font-size: 13px; opacity: 0.85; margin-bottom: 4px; }
    .metric-card .value { font-size: 28px; font-weight: 700; }
    .metric-card .delta { font-size: 12px; margin-top: 4px; opacity: 0.8; }
    .card-green  { background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%); }
    .card-blue   { background: linear-gradient(135deg, #2193b0 0%, #6dd5ed 100%); }
    .card-orange { background: linear-gradient(135deg, #f7971e 0%, #ffd200 100%); color: #333; }
    .card-red    { background: linear-gradient(135deg, #cb2d3e 0%, #ef473a 100%); }
    .section-header {
        font-size: 18px; font-weight: 700; color: #1a1a2e;
        border-left: 4px solid #667eea; padding-left: 10px;
        margin: 16px 0 12px;
    }
    .recommendation-box {
        background: #f0f9ff; border: 1px solid #bae6fd;
        border-radius: 8px; padding: 14px; margin: 8px 0;
    }
</style>
""", unsafe_allow_html=True)


def metric_card(label, value, delta="", color=""):
    st.markdown(f"""
    <div class="metric-card {color}">
        <div class="label">{label}</div>
        <div class="value">{value}</div>
        <div class="delta">{delta}</div>
    </div>""", unsafe_allow_html=True)


def fmt(n):
    if n >= 1_000_000:
        return f"${n/1_000_000:.2f}M"
    if n >= 1_000:
        return f"${n/1_000:.1f}K"
    return f"${n:.0f}"


# ─── Sidebar ────────────────────────────────────────────────────────────────

with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/artificial-intelligence.png", width=60)
    st.title("GenAI TCO Calculator")
    st.caption("Enterprise Multi-Cloud Edition")
    st.divider()

    st.markdown("### Company Profile")
    company_name = st.text_input("Company Name", "Acme Corp")
    industry = st.selectbox("Industry", [
        "Financial Services", "Healthcare", "Retail / E-commerce",
        "Technology", "Manufacturing", "Telecom", "Insurance", "Other"
    ])
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
        "Cloud API (Managed)", "Cloud VM (Self-managed)", "Self-hosted (On-Prem)"
    ])
    model_approach = st.selectbox("Model Approach", [
        "API / Third-party LLM", "Fine-tuned Open Source", "Self-hosted OSS"
    ])

    st.divider()
    st.caption("v1.0 · Built with Streamlit & Plotly")


# ─── Dynamic Inputs ─────────────────────────────────────────────────────────

# Collect all tab-specific inputs via session state defaults
if "infra_inputs" not in st.session_state:
    st.session_state.infra_inputs = {}

# ─── Tabs ───────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Dashboard",
    "💰 Cost Components",
    "🔄 Scenario Comparison",
    "🖥️ GPU Optimization",
    "📈 ROI & Payback",
    "📋 Assumptions & Docs",
])


# ════════════════════════════════════════════════════════════════════════════
# TAB 2 — Cost Components (compute first so results feed Dashboard)
# ════════════════════════════════════════════════════════════════════════════

with tab2:
    st.markdown('<div class="section-header">Infrastructure</div>', unsafe_allow_html=True)
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
        pricing_tier, storage_gb, cloud_provider, networking_egress_gb,
    )

    st.markdown('<div class="section-header">Model Cost</div>', unsafe_allow_html=True)
    ca, cb = st.columns(2)
    with ca:
        if model_approach == "API / Third-party LLM":
            selected_model = st.selectbox("LLM Model", list(LLM_API_PRICING.keys()), key="api_model")
            model_api_cost = compute_api_model_cost(
                selected_model, monthly_queries, avg_input_tokens, avg_output_tokens
            )
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
        if model_approach in ("API / Third-party LLM",) and "selected_model" in dir():
            p = LLM_API_PRICING.get(selected_model if model_approach == "API / Third-party LLM" else list(LLM_API_PRICING.keys())[0], {})
            rows = [
                ("Provider", p.get("provider", "—")),
                ("Input price", f"${p.get('input_per_1m', 0):.2f} / 1M tokens"),
                ("Output price", f"${p.get('output_per_1m', 0):.2f} / 1M tokens"),
                ("Context window", f"{p.get('context_window_k', '?')}K tokens"),
                ("Quality score", f"{p.get('quality_score', '?')} / 10"),
            ]
            st.table(pd.DataFrame(rows, columns=["Attribute", "Value"]).set_index("Attribute"))

    st.markdown('<div class="section-header">Data Pipeline</div>', unsafe_allow_html=True)
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
        vector_db_vectors_m, monthly_vector_queries_m, vector_db_type, cloud_provider,
    )

    st.markdown('<div class="section-header">Engineering & Labor</div>', unsafe_allow_html=True)
    e1, e2 = st.columns(2)
    with e1:
        team_comp = {}
        for role in LABOR_RATES:
            team_comp[role] = st.number_input(f"{role}", 0, 20, 1 if role in ("ML Engineer", "Data Engineer") else 0, key=f"hc_{role}")
    with e2:
        labor_alloc_pct = st.slider("% Time Allocated to GenAI", 10, 100, 70, key="labor_alloc")
        st.caption("Adjust if team is split across multiple projects")

    labor = compute_labor_cost(team_comp, labor_alloc_pct)

    st.markdown('<div class="section-header">Maintenance & Operations</div>', unsafe_allow_html=True)
    maintenance_pct = st.slider("Maintenance as % of Infra + Pipeline Cost", 5, 30, 15, key="maint_pct")
    maintenance = compute_maintenance_cost(infra["total"], pipeline["total"], maintenance_pct)

    # Total TCO
    tco = compute_total_tco(infra, model_api_cost, pipeline, labor, maintenance)

    # Cost waterfall chart
    st.markdown('<div class="section-header">Monthly Cost Breakdown</div>', unsafe_allow_html=True)
    breakdown = tco["breakdown"]
    fig_waterfall = go.Figure(go.Waterfall(
        name="Cost",
        orientation="v",
        measure=["relative"] * len(breakdown) + ["total"],
        x=list(breakdown.keys()) + ["Total TCO"],
        y=list(breakdown.values()) + [tco["monthly"]],
        text=[fmt(v) for v in list(breakdown.values()) + [tco["monthly"]]],
        textposition="outside",
        connector={"line": {"color": "rgb(63, 63, 63)"}},
        increasing={"marker": {"color": "#667eea"}},
        totals={"marker": {"color": "#2ecc71"}},
    ))
    fig_waterfall.update_layout(
        height=400, plot_bgcolor="#fafafa",
        margin=dict(t=20, b=20), showlegend=False,
        yaxis_title="USD / Month",
    )
    st.plotly_chart(fig_waterfall, use_container_width=True)

    # Infra detail table
    with st.expander("Infrastructure Detail"):
        rows = [
            ("GPU Compute", fmt(infra["gpu_compute"])),
            ("Storage", fmt(infra["storage"])),
            ("Networking", fmt(infra["networking"])),
            ("Cloud Overhead", fmt(infra["cloud_overhead"])),
        ]
        st.table(pd.DataFrame(rows, columns=["Component", "Monthly Cost"]).set_index("Component"))

    with st.expander("Data Pipeline Detail"):
        rows = [
            ("Ingestion", fmt(pipeline["ingestion"])),
            ("ETL / Transformation", fmt(pipeline["transformation"])),
            ("Vector DB Storage", fmt(pipeline["vector_db_storage"])),
            ("Vector DB Queries", fmt(pipeline["vector_db_queries"])),
            ("Raw Storage", fmt(pipeline["raw_storage"])),
        ]
        st.table(pd.DataFrame(rows, columns=["Component", "Monthly Cost"]).set_index("Component"))

    with st.expander("Labor Detail"):
        rows = [(role, fmt(cost)) for role, cost in labor["breakdown"].items() if cost > 0]
        if rows:
            st.table(pd.DataFrame(rows, columns=["Role", "Monthly Allocated"]).set_index("Role"))


# ════════════════════════════════════════════════════════════════════════════
# TAB 1 — Executive Dashboard
# ════════════════════════════════════════════════════════════════════════════

with tab1:
    st.markdown(f"## {company_name} · GenAI Investment Overview")
    st.caption(f"Use case: **{use_case}** · Cloud: **{cloud_provider}** · Scale: **{monthly_queries:,}** queries/month")
    st.divider()

    # KPI cards
    k1, k2, k3, k4 = st.columns(4)
    with k1:
        metric_card("Monthly TCO", fmt(tco["monthly"]), f"Annual: {fmt(tco['annual'])}", "")
    with k2:
        cost_per_query = tco["monthly"] / monthly_queries if monthly_queries else 0
        metric_card("Cost Per Query", f"${cost_per_query:.4f}", f"{monthly_queries:,} queries/mo", "card-blue")
    with k3:
        metric_card("Annual TCO", fmt(tco["annual"]), f"3-Year: {fmt(tco['three_year'])}", "card-orange")
    with k4:
        metric_card("Model Approach", model_approach.split(" ")[0], deployment_type, "card-green")

    st.divider()

    col_left, col_right = st.columns([1, 1])

    with col_left:
        st.markdown("#### Cost Distribution")
        fig_pie = px.pie(
            names=list(tco["breakdown"].keys()),
            values=list(tco["breakdown"].values()),
            color_discrete_sequence=px.colors.qualitative.Bold,
            hole=0.45,
        )
        fig_pie.update_traces(textposition="outside", textinfo="label+percent")
        fig_pie.update_layout(height=380, margin=dict(t=10, b=10), showlegend=True)
        st.plotly_chart(fig_pie, use_container_width=True)

    with col_right:
        st.markdown("#### 3-Year Cumulative TCO")
        proj = project_costs_over_time(tco["monthly"], 36)
        df_proj = pd.DataFrame(proj)
        fig_tco = go.Figure()
        fig_tco.add_trace(go.Scatter(
            x=df_proj["month"], y=df_proj["cumulative"],
            fill="tozeroy", name="Cumulative Cost",
            line=dict(color="#667eea", width=2),
            fillcolor="rgba(102,126,234,0.15)",
        ))
        fig_tco.add_trace(go.Bar(
            x=df_proj["month"], y=df_proj["monthly"],
            name="Monthly Cost", marker_color="rgba(118,75,162,0.4)",
            yaxis="y2",
        ))
        fig_tco.update_layout(
            height=380, plot_bgcolor="#fafafa",
            margin=dict(t=10, b=10),
            yaxis=dict(title="Cumulative ($)", side="left"),
            yaxis2=dict(title="Monthly ($)", side="right", overlaying="y"),
            legend=dict(orientation="h", y=-0.15),
        )
        st.plotly_chart(fig_tco, use_container_width=True)

    # Summary table
    st.markdown("#### Cost Summary Table")
    summary_data = {
        "Period": ["Monthly", "Quarterly", "Annual", "3-Year", "5-Year"],
        "TCO ($)": [
            f"${tco['monthly']:,.0f}",
            f"${tco['monthly'] * 3:,.0f}",
            f"${tco['annual']:,.0f}",
            f"${tco['three_year']:,.0f}",
            f"${tco['five_year']:,.0f}",
        ],
        "Cost / Query ($)": [f"${tco['monthly'] / monthly_queries:.5f}"] * 5 if monthly_queries else ["—"] * 5,
        "Largest Cost Driver": [max(tco["breakdown"], key=tco["breakdown"].get)] * 5,
    }
    st.dataframe(pd.DataFrame(summary_data), use_container_width=True, hide_index=True)

    # Recommendations
    st.markdown("#### Strategic Recommendations")
    recs = []
    largest_driver = max(tco["breakdown"], key=tco["breakdown"].get)
    if largest_driver == "Labor":
        recs.append(("🧑‍💼 Labor Dominates Costs", "Consider increasing automation and reducing manual MLOps overhead. Evaluate managed services to reduce engineering burden."))
    if largest_driver == "Model / API":
        recs.append(("💡 Optimize Model Selection", f"API spend is your top cost. Consider switching to a lighter model (Claude Haiku, Gemini Flash) for routine queries, reserving premium models for complex tasks."))
    if infra["gpu_compute"] > 0 and utilization_pct < 40:
        recs.append(("🖥️ Low GPU Utilization", f"GPUs running at {utilization_pct}% utilization. Use spot/preemptible instances for batch workloads and scale down idle capacity."))
    if deployment_type == "Cloud API (Managed)":
        recs.append(("☁️ Evaluate Reserved Pricing", "Switching to 1-year reserved GPU instances can save 30–45% vs on-demand pricing at your usage scale."))
    recs.append(("📊 Monitor & Iterate", "Set monthly TCO alerts and query cost dashboards. Even a 10% reduction in token usage can translate to significant savings at scale."))

    for title, body in recs:
        st.markdown(f"""<div class="recommendation-box"><strong>{title}</strong><br><span style="font-size:13px; color:#444">{body}</span></div>""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 3 — Scenario Comparison
# ════════════════════════════════════════════════════════════════════════════

with tab3:
    st.markdown("### Build vs Buy · Cloud vs On-Prem · Model Alternatives")

    base_inputs = {
        "monthly_queries": monthly_queries,
        "avg_input_tokens": avg_input_tokens,
        "avg_output_tokens": avg_output_tokens,
        "storage_gb": storage_gb,
        "networking_egress_gb": networking_egress_gb,
        "utilization_pct": utilization_pct,
    }
    scenarios = compute_scenario_comparison(base_inputs)

    # Bar chart comparison
    fig_sc = go.Figure()
    colors = ["#667eea", "#2ecc71", "#e74c3c", "#f39c12"]
    for i, (name, vals) in enumerate(scenarios.items()):
        fig_sc.add_trace(go.Bar(
            name=name,
            x=["Model / Infra Cost", "Infrastructure Cost", "Total Monthly"],
            y=[vals["model"], vals["infra"], vals["total"]],
            marker_color=colors[i % len(colors)],
            text=[fmt(vals["model"]), fmt(vals["infra"]), fmt(vals["total"])],
            textposition="outside",
        ))
    fig_sc.update_layout(
        barmode="group", height=420, plot_bgcolor="#fafafa",
        yaxis_title="USD / Month", margin=dict(t=20),
        legend=dict(orientation="h", y=-0.2),
    )
    st.plotly_chart(fig_sc, use_container_width=True)

    # Scenario table
    st.markdown("#### Detailed Scenario Breakdown")
    lowest = min(scenarios, key=lambda k: scenarios[k]["total"])
    rows = []
    for name, vals in scenarios.items():
        savings_vs_current = tco["monthly"] - vals["total"]
        rows.append({
            "Scenario": name,
            "Model Cost": fmt(vals["model"]),
            "Infra Cost": fmt(vals["infra"]),
            "Total/Month": fmt(vals["total"]),
            "Annual": fmt(vals["total"] * 12),
            "vs Current": f"+{fmt(savings_vs_current)}" if savings_vs_current > 0 else fmt(savings_vs_current),
            "Best?": "✅ Lowest cost" if name == lowest else "",
        })
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    st.divider()
    st.markdown("### Model-by-Model Cost Comparison")
    st.caption(f"At **{monthly_queries:,}** monthly queries · {avg_input_tokens} input + {avg_output_tokens} output tokens")

    model_rows = []
    for mname, mp in LLM_API_PRICING.items():
        cost = compute_api_model_cost(mname, monthly_queries, avg_input_tokens, avg_output_tokens)
        model_rows.append({
            "Model": mname,
            "Provider": mp["provider"],
            "Monthly Cost": fmt(cost),
            "Annual Cost": fmt(cost * 12),
            "Input ($/1M)": f"${mp['input_per_1m']}",
            "Output ($/1M)": f"${mp['output_per_1m']}",
            "Context": f"{mp['context_window_k']}K",
            "Quality": "⭐" * mp["quality_score"],
            "_sort": cost,
        })
    df_models = pd.DataFrame(model_rows).sort_values("_sort").drop(columns=["_sort"])
    st.dataframe(df_models, use_container_width=True, hide_index=True)

    # 3-year cloud vs on-prem projection
    st.divider()
    st.markdown("### 3-Year Cloud vs On-Prem Projection")
    cloud_monthly = list(scenarios.values())[0]["total"] + labor["total"] + pipeline["total"]
    onprem_monthly = list(scenarios.values())[2]["total"] + labor["total"] + pipeline["total"]
    onprem_capex = GPU_PRICING.get(gpu_type, {}).get("on_demand_hourly", 3.67) * gpu_count * 8760 * 0.2

    months_range = list(range(1, 37))
    cloud_cum = [cloud_monthly * m for m in months_range]
    onprem_cum = [onprem_capex + onprem_monthly * m for m in months_range]

    fig_cvo = go.Figure()
    fig_cvo.add_trace(go.Scatter(x=months_range, y=cloud_cum, name="Cloud API", line=dict(color="#667eea", width=2)))
    fig_cvo.add_trace(go.Scatter(x=months_range, y=onprem_cum, name="On-Prem (incl. CapEx)", line=dict(color="#e74c3c", width=2)))
    crossover = next((m for m, (c, o) in enumerate(zip(cloud_cum, onprem_cum), 1) if c > o), None)
    if crossover:
        fig_cvo.add_vline(x=crossover, line_dash="dash", line_color="green",
                          annotation_text=f"Crossover: Month {crossover}", annotation_position="top right")
    fig_cvo.update_layout(height=380, plot_bgcolor="#fafafa", yaxis_title="Cumulative Cost ($)",
                          xaxis_title="Month", margin=dict(t=20))
    st.plotly_chart(fig_cvo, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 4 — GPU Optimization
# ════════════════════════════════════════════════════════════════════════════

with tab4:
    st.markdown("### GPU Cost Optimization Analysis")

    g1, g2 = st.columns(2)
    with g1:
        gpu_opt_type = st.selectbox("Select GPU for Analysis", list(GPU_PRICING.keys()), key="gpu_opt")
        gpu_opt_count = st.slider("GPU Count", 1, 32, 4, key="gpu_opt_count")
    with g2:
        util_range = st.slider("Utilization Range (%)", 10, 100, (20, 90), key="util_range")

    gpu_info = GPU_PRICING[gpu_opt_type]

    # GPU specs card
    st.markdown("#### GPU Specifications")
    sp1, sp2, sp3, sp4 = st.columns(4)
    sp1.metric("Memory", f"{gpu_info['memory_gb']} GB")
    sp2.metric("BF16 Performance", f"{gpu_info['flops_bf16_tflops']} TFLOPS")
    sp3.metric("On-Demand Rate", f"${gpu_info['on_demand_hourly']:.2f}/hr")
    sp4.metric("Reserved 1yr Rate", f"${gpu_info['reserved_1yr_hourly']:.2f}/hr")

    # Utilization vs Cost curves
    util_points = list(range(10, 101, 5))
    data = []
    for tier, label in [("on_demand", "On-Demand"), ("reserved_1yr", "Reserved 1yr"), ("reserved_3yr", "Reserved 3yr")]:
        for u in util_points:
            data.append({
                "Utilization (%)": u,
                "Monthly Cost ($)": compute_gpu_monthly_cost(gpu_opt_type, gpu_opt_count, u, tier),
                "Tier": label,
            })
    df_util = pd.DataFrame(data)
    fig_util = px.line(df_util, x="Utilization (%)", y="Monthly Cost ($)", color="Tier",
                       color_discrete_sequence=["#e74c3c", "#667eea", "#2ecc71"],
                       markers=True)
    fig_util.update_layout(height=380, plot_bgcolor="#fafafa", margin=dict(t=20))
    st.plotly_chart(fig_util, use_container_width=True)

    # GPU comparison table
    st.markdown("#### All GPU Types — Side-by-Side at Current Utilization")
    gpu_rows = []
    for gname, gdata in GPU_PRICING.items():
        od = compute_gpu_monthly_cost(gname, gpu_opt_count, utilization_pct, "on_demand")
        r1 = compute_gpu_monthly_cost(gname, gpu_opt_count, utilization_pct, "reserved_1yr")
        r3 = compute_gpu_monthly_cost(gname, gpu_opt_count, utilization_pct, "reserved_3yr")
        savings_1yr = (1 - r1 / od) * 100 if od else 0
        perf_per_dollar = gdata["flops_bf16_tflops"] / gdata["on_demand_hourly"]
        gpu_rows.append({
            "GPU": gname,
            "Memory": f"{gdata['memory_gb']} GB",
            "TFLOPS (BF16)": gdata["flops_bf16_tflops"],
            "On-Demand/mo": fmt(od),
            "Reserved 1yr/mo": fmt(r1),
            "Reserved 3yr/mo": fmt(r3),
            "Savings (1yr)": f"{savings_1yr:.0f}%",
            "TFLOPS/$": f"{perf_per_dollar:.1f}",
        })
    st.dataframe(pd.DataFrame(gpu_rows), use_container_width=True, hide_index=True)

    # Idle cost impact
    st.markdown("#### Impact of Idle GPU Time")
    idle_data = []
    for u in range(10, 101, 10):
        od = compute_gpu_monthly_cost(gpu_opt_type, gpu_opt_count, u, "on_demand")
        idle_data.append({"Utilization (%)": u, "Cost ($)": od, "Wasted ($)": od * (1 - u / 100) * 0.5})
    df_idle = pd.DataFrame(idle_data)
    fig_idle = go.Figure()
    fig_idle.add_trace(go.Bar(x=df_idle["Utilization (%)"], y=df_idle["Cost ($)"], name="Active Cost", marker_color="#667eea"))
    fig_idle.add_trace(go.Bar(x=df_idle["Utilization (%)"], y=df_idle["Wasted ($)"], name="Wasted (Idle)", marker_color="#e74c3c"))
    fig_idle.update_layout(barmode="overlay", height=340, plot_bgcolor="#fafafa",
                           yaxis_title="Monthly Cost ($)", margin=dict(t=20))
    st.plotly_chart(fig_idle, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 5 — ROI & Payback
# ════════════════════════════════════════════════════════════════════════════

with tab5:
    st.markdown("### ROI & Business Case Analysis")

    st.markdown('<div class="section-header">Legacy System Baseline</div>', unsafe_allow_html=True)
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        legacy_type = st.selectbox("Legacy System Type", list(LEGACY_SYSTEM_BENCHMARKS.keys()), key="legacy_type")
        legacy_units = st.number_input("Number of Units (agents / seats / licenses)", 1, 10_000, 50, key="legacy_units")
        legacy_monthly = LEGACY_SYSTEM_BENCHMARKS[legacy_type] * legacy_units / 12
        st.metric("Estimated Legacy Monthly Cost", fmt(legacy_monthly))
    with r1c2:
        initial_investment = st.number_input("Initial Setup Investment ($)", 0, 5_000_000, 50_000, step=5000, key="initial_inv")
        st.caption("Includes: integration, data prep, training, infra setup, change management")

    st.markdown('<div class="section-header">Productivity Gains</div>', unsafe_allow_html=True)
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        employees_affected = st.number_input("Employees Affected", 1, 10_000, 100, key="emp_aff")
    with p2:
        hours_saved_monthly = st.slider("Hours Saved / Employee / Month", 0, 80, 15, key="hrs_saved")
    with p3:
        avg_hourly_rate = st.number_input("Avg Hourly Rate ($)", 10, 200, 45, key="hrly_rate")
    with p4:
        automation_rate_pct = st.slider("Automation Rate %", 0, 100, 40, key="auto_rate")

    prod_gains = compute_productivity_gains(employees_affected, hours_saved_monthly, avg_hourly_rate, automation_rate_pct)

    st.markdown('<div class="section-header">Cost Avoidance</div>', unsafe_allow_html=True)
    ca1, ca2, ca3 = st.columns(3)
    with ca1:
        headcount_reduction = st.number_input("Headcount Reduction (FTEs)", 0, 500, 10, key="hc_red")
        avg_agent_salary = st.number_input("Avg Agent Annual Salary ($)", 20_000, 200_000, 65_000, step=5000, key="agent_sal")
    with ca2:
        error_reduction_pct = st.slider("Error Reduction %", 0, 100, 60, key="err_red")
        avg_error_cost = st.number_input("Avg Cost Per Error ($)", 0, 10_000, 200, key="err_cost")
    with ca3:
        monthly_error_volume = st.number_input("Monthly Error Volume", 0, 100_000, 500, key="err_vol")

    cost_avoid = compute_cost_avoidance(
        legacy_monthly, headcount_reduction, avg_agent_salary,
        error_reduction_pct, avg_error_cost, monthly_error_volume,
    )

    st.markdown('<div class="section-header">Revenue Impact</div>', unsafe_allow_html=True)
    rv1, rv2, rv3 = st.columns(3)
    with rv1:
        monthly_interactions = st.number_input("Monthly Customer Interactions", 100, 10_000_000, 50_000, step=1000, key="cust_int")
        avg_rev_per_customer = st.number_input("Avg Revenue Per Customer ($)", 1, 10_000, 120, key="avg_rev")
    with rv2:
        baseline_csat = st.slider("Baseline CSAT Score", 1, 10, 6, key="base_csat")
        target_csat = st.slider("Target CSAT Score", 1, 10, 8, key="tgt_csat")
    with rv3:
        churn_reduction_pct = st.slider("Churn Reduction %", 0, 50, 10, key="churn_red")
        upsell_lift_pct = st.slider("Upsell Lift %", 0, 20, 5, key="upsell_lift")

    rev_impact = compute_revenue_impact(
        monthly_interactions, baseline_csat, target_csat,
        avg_rev_per_customer, churn_reduction_pct, upsell_lift_pct,
    )

    roi_result = compute_full_roi(
        tco["monthly"], prod_gains, cost_avoid, rev_impact, initial_investment, months=60
    )

    # ROI KPIs
    st.divider()
    kp1, kp2, kp3, kp4 = st.columns(4)
    with kp1:
        payback = roi_result["payback_months"]
        payback_str = f"{payback:.1f} months" if payback != float("inf") else "Never"
        metric_card("Payback Period", payback_str, "Based on net monthly benefit", "card-blue")
    with kp2:
        metric_card("Monthly Net Benefit", fmt(roi_result["monthly_net"]),
                    "Benefits minus TCO", "card-green" if roi_result["monthly_net"] > 0 else "card-red")
    with kp3:
        npv_str = fmt(abs(roi_result["npv"])) + (" ✅" if roi_result["npv"] > 0 else " ⚠️")
        metric_card("3-Year NPV", npv_str, "At 10% discount rate", "card-green" if roi_result["npv"] > 0 else "")
    with kp4:
        irr_str = f"{roi_result['irr_pct']:.1f}%" if roi_result["irr_pct"] else "—"
        metric_card("IRR", irr_str, "Internal Rate of Return", "card-orange")

    # Benefits breakdown
    col_ben, col_time = st.columns(2)
    with col_ben:
        st.markdown("#### Monthly Benefits Breakdown")
        fig_ben = px.pie(
            names=list(roi_result["benefit_breakdown"].keys()),
            values=list(roi_result["benefit_breakdown"].values()),
            color_discrete_sequence=["#2ecc71", "#3498db", "#9b59b6"],
            hole=0.4,
        )
        fig_ben.update_traces(textposition="outside", textinfo="label+percent+value",
                              texttemplate="%{label}<br>%{percent}<br>$%{value:,.0f}")
        fig_ben.update_layout(height=360, margin=dict(t=10, b=30), showlegend=False)
        st.plotly_chart(fig_ben, use_container_width=True)

    with col_time:
        st.markdown("#### 5-Year Cumulative Value")
        tl = roi_result["timeline"]
        df_tl = pd.DataFrame(tl)
        fig_roi = go.Figure()
        fig_roi.add_trace(go.Scatter(
            x=df_tl["month"], y=df_tl["cumulative_benefit"],
            name="Cumulative Benefit", fill="tozeroy",
            line=dict(color="#2ecc71", width=2), fillcolor="rgba(46,204,113,0.15)",
        ))
        fig_roi.add_trace(go.Scatter(
            x=df_tl["month"], y=df_tl["cumulative_cost"],
            name="Cumulative Cost", fill="tozeroy",
            line=dict(color="#e74c3c", width=2), fillcolor="rgba(231,76,60,0.1)",
        ))
        # Payback line
        if payback != float("inf") and payback <= 60:
            fig_roi.add_vline(x=payback, line_dash="dash", line_color="#f39c12",
                              annotation_text=f"Payback: M{payback:.0f}", annotation_position="top left")
        fig_roi.update_layout(height=360, plot_bgcolor="#fafafa",
                              yaxis_title="Cumulative Value ($)", xaxis_title="Month",
                              margin=dict(t=10), legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(fig_roi, use_container_width=True)

    # ROI over time
    st.markdown("#### ROI % Over Time")
    fig_roi_pct = go.Figure()
    fig_roi_pct.add_trace(go.Scatter(
        x=df_tl["month"], y=df_tl["roi_pct"],
        line=dict(color="#9b59b6", width=2),
        fill="tozeroy", fillcolor="rgba(155,89,182,0.1)",
    ))
    fig_roi_pct.add_hline(y=0, line_dash="dash", line_color="gray")
    fig_roi_pct.add_hline(y=100, line_dash="dot", line_color="#2ecc71", annotation_text="100% ROI")
    fig_roi_pct.update_layout(height=280, plot_bgcolor="#fafafa",
                               yaxis_title="ROI (%)", xaxis_title="Month", margin=dict(t=10))
    st.plotly_chart(fig_roi_pct, use_container_width=True)


# ════════════════════════════════════════════════════════════════════════════
# TAB 6 — Assumptions & Documentation
# ════════════════════════════════════════════════════════════════════════════

with tab6:
    st.markdown("### Assumptions, Formulas & Usage Guide")

    with st.expander("📐 Cost Formula Reference", expanded=True):
        st.markdown("""
**Infrastructure**
```
GPU Monthly Cost = GPU_Count × GPU_Hours/Month × Hourly_Rate × Utilization%
                 + GPU_Count × GPU_Hours/Month × Idle_Rate × (1 - Utilization%)
Storage Cost     = Storage_GB × Rate_per_GB_Month
Network Cost     = Egress_GB × Rate_per_GB
```

**Model / API**
```
API Cost = (Monthly_Queries × Avg_Input_Tokens  / 1,000,000 × Input_Rate)
         + (Monthly_Queries × Avg_Output_Tokens / 1,000,000 × Output_Rate)
```

**Data Pipeline**
```
Pipeline Cost = Ingestion_Cost + ETL_Compute + Vector_DB_Storage + Vector_Queries + Raw_Storage
```

**Labor**
```
Monthly Labor = Σ (Headcount × Annual_Salary / 12 × Allocation%)
```

**Total Monthly TCO**
```
TCO = Infrastructure + Model_API + Data_Pipeline + Labor + Maintenance
```

**ROI**
```
Monthly Net = Total_Monthly_Benefits - Monthly_TCO
Payback     = Initial_Investment / Monthly_Net
NPV         = -Initial_Investment + Σ (Net_Cashflow_t / (1 + r)^t)
```
        """)

    with st.expander("💡 Pricing Assumptions"):
        st.markdown("#### GPU Pricing (GCP / AWS / Azure reference rates)")
        gpu_df = pd.DataFrame([
            {
                "GPU": k,
                "On-Demand $/hr": v["on_demand_hourly"],
                "Reserved 1yr $/hr": v["reserved_1yr_hourly"],
                "Reserved 3yr $/hr": v["reserved_3yr_hourly"],
                "Memory (GB)": v["memory_gb"],
                "BF16 TFLOPS": v["flops_bf16_tflops"],
            }
            for k, v in GPU_PRICING.items()
        ])
        st.dataframe(gpu_df, use_container_width=True, hide_index=True)

        st.markdown("#### LLM API Pricing")
        llm_df = pd.DataFrame([
            {
                "Model": k,
                "Provider": v["provider"],
                "Input $/1M": v["input_per_1m"],
                "Output $/1M": v["output_per_1m"],
                "Context Window": f"{v['context_window_k']}K",
            }
            for k, v in LLM_API_PRICING.items()
        ])
        st.dataframe(llm_df, use_container_width=True, hide_index=True)

    with st.expander("🏗️ Architecture Notes"):
        st.markdown("""
**Model Deployment Approaches**

| Approach | Pros | Cons | Best For |
|---|---|---|---|
| Cloud API (Managed) | Zero infra overhead, instant scale | Per-token costs grow with volume, data privacy | Early stage, low-medium volume |
| Fine-tuned OSS | Domain accuracy, lower inference cost long-term | Upfront training cost, MLOps overhead | Specialized tasks, >1M queries/month |
| Self-hosted On-Prem | Full control, no token costs, data sovereignty | High CapEx, GPU ops team required | Regulated industries, very high volume |

**Data Pipeline Architecture**
- **Ingestion Layer**: Kafka / Pub/Sub for streaming; batch ETL for historical data
- **Processing Layer**: Apache Beam / Dataflow for transformation
- **Vector Store**: Embeddings stored in Pinecone / Weaviate / Qdrant
- **Storage**: Cloud Storage (raw) + BigQuery (analytics)
- **Caching**: Redis for frequent query results (reduces API costs 20–40%)

**Cost Optimization Levers**
1. Semantic caching — cache similar query responses
2. Token compression — strip whitespace, use shorter prompts
3. Model routing — lightweight model for simple queries, premium for complex
4. Batch processing — group non-realtime requests for cheaper batch APIs
5. Reserved capacity — commit to 1yr for 35–45% savings
        """)

    with st.expander("📊 Export Current Scenario"):
        export_data = {
            "Company": company_name,
            "Industry": industry,
            "Use Case": use_case,
            "Monthly Queries": monthly_queries,
            "Cloud Provider": cloud_provider,
            "Deployment": deployment_type,
            "Model Approach": model_approach,
            "Monthly TCO": tco["monthly"],
            "Annual TCO": tco["annual"],
            "3-Year TCO": tco["three_year"],
            "Infrastructure Cost": infra["total"],
            "Model/API Cost": model_api_cost,
            "Data Pipeline Cost": pipeline["total"],
            "Labor Cost": labor["total"],
            "Maintenance Cost": maintenance,
            "Monthly Benefits": roi_result["monthly_benefits"],
            "Payback Months": roi_result["payback_months"] if roi_result["payback_months"] != float("inf") else "N/A",
            "NPV": roi_result["npv"],
        }
        df_export = pd.DataFrame([export_data]).T.reset_index()
        df_export.columns = ["Metric", "Value"]
        st.dataframe(df_export, use_container_width=True, hide_index=True)
        csv = df_export.to_csv(index=False)
        st.download_button("Download as CSV", csv, f"genai_tco_{company_name.replace(' ', '_')}.csv", "text/csv")
