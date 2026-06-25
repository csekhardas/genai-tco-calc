# 🤖 Enterprise GenAI TCO & ROI Calculator

> **Multi-cloud, model-agnostic Total Cost of Ownership and Return on Investment calculator for enterprise GenAI deployments.**

Built for finance, engineering, and product teams who need executive-ready numbers — not back-of-napkin estimates.

---

## 🖥️ Live Demo

```bash
git clone https://github.com/csekhardas/genai-tco-calc.git
cd genai-tco-calc
pip install -r requirements.txt
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501)

---

## 📸 What's Inside

| Tab | What You Get |
|-----|-------------|
| 📊 **Executive Dashboard** | KPI cards, cost donut, 3-year TCO projection, smart recommendations |
| 💰 **Cost Components** | Waterfall breakdown across infra, model API, data pipeline, labor, maintenance |
| 🔄 **Scenario Comparison** | Cloud API vs Fine-tuned vs Self-hosted vs On-Prem — all side by side |
| 🖥️ **GPU Optimization** | Utilization curves, idle-waste analysis, reserved vs on-demand savings |
| 📈 **ROI & Payback** | Payback period, NPV, IRR, 5-year cumulative benefit/cost chart |
| 📋 **Assumptions & Docs** | Formula reference, pricing tables, architecture notes, CSV export |

---

## 🏗️ Architecture

```
genai-tco-calc/
├── app.py                  # Streamlit UI — 6 tabs, sidebar inputs, all charts
├── models/
│   ├── pricing_data.py     # GPU, LLM API, vector DB, labor pricing constants
│   ├── cost_model.py       # TCO computation engine
│   └── roi_model.py        # ROI, NPV, IRR, payback calculations
└── requirements.txt
```

---

## 💡 Key Features

### Multi-Cloud Pricing Engine
Real-world rates for **GCP, AWS, and Azure** — GPU compute, storage, egress, and managed services — refreshable from cloud pricing APIs.

### 5 GPU Types Modeled
| GPU | Memory | BF16 TFLOPS | On-Demand/hr |
|-----|--------|-------------|--------------|
| NVIDIA H100 (80GB) | 80 GB | 989 | $8.00 |
| NVIDIA A100 (80GB) | 80 GB | 312 | $3.67 |
| TPU v4 | 32 GB | 275 | $3.22 |
| NVIDIA V100 | 16 GB | 112 | $2.48 |
| NVIDIA L4 | 24 GB | 121.5 | $0.75 |

On-demand · Reserved 1yr · Reserved 3yr pricing tiers for all.

### 8 LLM APIs Compared
GPT-4 Turbo, GPT-4o, GPT-3.5 Turbo, Claude 3 Opus/Sonnet/Haiku, Gemini 1.5 Pro/Flash — cost computed live at your query volume.

### 3 Deployment Scenarios
- **Cloud API** — pay-per-token, zero infra overhead
- **Fine-tuned Open Source** — one-time training cost + inference GPU
- **Self-hosted On-Prem** — full GPU cluster, no token costs, data sovereignty

### Full ROI Model
```
Monthly Net Benefit  =  Productivity Gains + Cost Avoidance + Revenue Impact − TCO
Payback Period       =  Initial Investment ÷ Monthly Net Benefit
NPV                  =  −CapEx + Σ (Net Cashflow_t / (1 + r)^t)   [r = 10% default]
IRR                  =  Bisection solve for monthly rate where NPV = 0
```

---

## 🚀 Quick Start

### Prerequisites
- Python 3.9+
- pip

### Install & Run
```bash
pip install -r requirements.txt
streamlit run app.py
```

### Configure Your Scenario (Sidebar)
1. **Company Profile** — name, industry, headcount
2. **Use Case** — customer support, knowledge base, document processing, etc.
3. **Scale** — monthly queries, avg input/output tokens
4. **Deployment** — cloud provider, deployment type, model approach

Everything else updates live across all tabs.

---

## 📐 Cost Formula Reference

**Infrastructure**
```
GPU Monthly = GPU_Count × 730hrs × Rate × Utilization%
            + GPU_Count × 730hrs × IdleRate × (1 − Utilization%)
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

---

## 🏢 Use Cases

| Team | How They Use It |
|------|----------------|
| **CFO / Finance** | 3-year TCO projections, payback period, NPV for capital approval |
| **Engineering** | GPU type selection, reserved vs on-demand savings, utilization optimization |
| **Product / Strategy** | Build vs buy decisions, cloud vs on-prem tradeoffs |
| **Sales / Pre-sales** | Client-facing ROI models for GenAI adoption proposals |
| **MLOps** | Data pipeline cost modeling, vector DB sizing |

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| UI | [Streamlit](https://streamlit.io) |
| Charts | [Plotly](https://plotly.com) |
| Data | [Pandas](https://pandas.pydata.org) · [NumPy](https://numpy.org) |
| Pricing | GCP · AWS · Azure reference rates (June 2025) |

---

## 📊 Sample Output

For a mid-scale deployment (500K queries/month, GPT-4o, 4× A100, 5-person ML team):

| Metric | Value |
|--------|-------|
| Monthly TCO | ~$40,000 |
| Annual TCO | ~$480,000 |
| Cost per Query | $0.00008 |
| Payback Period | 2–4 months |
| 3-Year NPV | $2.1M+ |

---

## 🔄 Roadmap

- [ ] Live cloud pricing API integration (GCP Billing, AWS Pricing API)
- [ ] BigQuery backend for usage simulation at scale
- [ ] Multi-tenant SaaS cost modeling
- [ ] Automated scenario PDF report export
- [ ] Streamlit Cloud one-click deploy button

---

## 📄 License

MIT — free to use, modify, and distribute.

---

<div align="center">
  <sub>Built with ❤️ using Streamlit · Plotly · Python</sub>
</div>
