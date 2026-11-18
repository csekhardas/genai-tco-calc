<div align="center">
  <img src="AI-CostLens_logo.png" alt="AI CostLens" width="260" />
  <h3>GenAI TCO & ROI Calculator</h3>
  <p>Enterprise-grade total cost of ownership and return on investment analysis for AI deployments.</p>

  [![Live App](https://img.shields.io/badge/Live%20App-ai--costlens--tco--calc.streamlit.app-FF4B4B?style=flat&logo=streamlit)](https://ai-costlens-tco-calc.streamlit.app/)
  [![Python](https://img.shields.io/badge/Python-3.9%2B-3776AB?style=flat&logo=python&logoColor=white)](https://python.org)
  [![License: MIT](https://img.shields.io/badge/License-MIT-22C55E?style=flat)](LICENSE)
</div>

---

## Overview

**AI CostLens** helps finance, engineering, and product teams build accurate business cases for GenAI initiatives. Configure your scenario — model, scale, infrastructure, team — and get executive-ready numbers across nine analysis dimensions in seconds.

No spreadsheets. No guesswork. No stale vendor quotes.

---

## Live App

[**ai-costlens-tco-calc.streamlit.app**](https://ai-costlens-tco-calc.streamlit.app/)

---

## What's Inside

| Tab | Description |
|-----|-------------|
| **Executive Dashboard** | KPI summary, cost distribution, smart optimization recommendations |
| **Cost Breakdown** | Itemized monthly spend — infrastructure, API, data pipeline, labor, maintenance |
| **Scenario Comparison** | Cloud API vs. Fine-tuned vs. Self-hosted vs. Hybrid — all side by side |
| **GPU Optimization** | Utilization curves, idle-waste analysis, reserved vs. on-demand savings |
| **ROI & Payback** | Payback period, NPV, IRR, 5-year cumulative cash flow chart |
| **Tokenomics** | Token volume analysis, semantic caching savings, 12-month cost projection |
| **Model Catalog** | Side-by-side comparison of all 52 supported models at your query volume |
| **Governance** | Compliance, security, and audit cost modeling by framework |
| **Export** | Download full report as CSV, JSON, or Markdown |

---

## Coverage

### 52 LLM Models across 9 Providers

| Provider | Models |
|----------|--------|
| **OpenAI** | GPT-4o, GPT-4.1 family, GPT-4.5 Turbo, o1, o3, o3-pro, o4-mini |
| **Anthropic** | Claude Fable 5, Opus 4.6–4.8, Sonnet 4.6, Haiku 4.5, Claude 3.5 family |
| **Google** | Gemini 2.5 Ultra/Pro/Flash, 2.0 Flash, 1.5 Pro/Flash |
| **xAI** | Grok 3, Grok 3 mini, Grok 2 |
| **DeepSeek** | DeepSeek V3, R1, R1 Zero |
| **Mistral** | Large 2, Small 3.1, Codestral, Mixtral 8x22B |
| **Cohere** | Command A, Command R+, Command R |
| **Amazon** | Nova Premier, Nova Pro, Nova Lite, Nova Micro |
| **Meta** | Llama 4 Scout, Llama 4 Maverick, Llama 3.3 70B |

### 28 GPU & Accelerator Types

| Family | Models |
|--------|--------|
| **NVIDIA Blackwell** | GB200, B200, B100 |
| **NVIDIA Grace-Hopper** | GH200 96GB |
| **NVIDIA Hopper** | H200 141GB, H100 SXM/PCIe/NVL |
| **NVIDIA Ampere / Ada** | A100 80/40GB, A10G, L40S, L4 |
| **AMD Instinct** | MI325X 288GB, MI300X 192GB, MI250X 128GB |
| **Intel Gaudi** | Gaudi 3 128GB, Gaudi 2 96GB |
| **AWS Trainium** | Trainium 2, Trainium 1 |
| **Google TPU** | v6e Trillium, v5p, v5e, v4 |

### 23 Open-Source Self-Hosted Models
Llama 4/3 family, DeepSeek V3/R1 distill variants, Gemma 3, Qwen 2.5, Mistral/Mixtral, Phi-4, and more — with GPU memory and throughput estimates for capacity planning.

---

## Deployment Scenarios Modeled

- **Cloud API** — pay-per-token, zero infrastructure overhead
- **Fine-tuned Open Source** — one-time training cost + dedicated inference GPU
- **Self-hosted On-Prem** — full GPU cluster with data sovereignty
- **Hybrid** — blended approach with routing logic

---

## Who Uses It

| Audience | How They Use It |
|----------|----------------|
| **CFO / Finance** | 3-year TCO projections and payback period for capital approval |
| **Engineering / MLOps** | GPU selection, utilization optimization, reserved pricing analysis |
| **Product / Strategy** | Build vs. buy decisions, cloud vs. on-prem trade-offs |
| **Sales / Pre-sales** | Client-facing ROI models for GenAI adoption proposals |
| **Compliance / Legal** | Governance and security cost modeling by framework (SOC 2, HIPAA, GDPR) |

---

## Quick Start

**Prerequisites:** Python 3.9+

```bash
git clone https://github.com/csekhardas/genai-tco-calc.git
cd genai-tco-calc
pip install -r requirements.txt
streamlit run app.py
```

App opens at `http://localhost:8501`.

---

## Project Structure

```
genai-tco-calc/
├── app.py                  # Streamlit UI — 9 tabs, sidebar inputs, charts
├── models/
│   ├── pricing_data.py     # 52 LLM models, 28 GPUs, labor & vector DB rates
│   ├── cost_model.py       # TCO computation engine
│   └── roi_model.py        # ROI, NPV, IRR, payback calculations
├── AI-CostLens_logo.png
└── requirements.txt
```

---

## Tech Stack

| Layer | Technology |
|-------|------------|
| UI | [Streamlit](https://streamlit.io) |
| Charts | [Plotly](https://plotly.com) |
| Data | [Pandas](https://pandas.pydata.org) |
| Pricing | GCP · AWS · Azure · provider reference rates (Nov 2026) |

---

## License

MIT — free to use, modify, and distribute.

---

<div align="center">
  <sub>Built with Streamlit · Plotly · Python</sub>
</div>
