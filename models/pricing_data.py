GPU_PRICING = {
    "NVIDIA A100 (80GB)": {
        "on_demand_hourly": 3.67,
        "reserved_1yr_hourly": 2.20,
        "reserved_3yr_hourly": 1.50,
        "memory_gb": 80,
        "flops_bf16_tflops": 312,
        "cloud": "GCP / AWS / Azure",
    },
    "NVIDIA H100 (80GB)": {
        "on_demand_hourly": 8.00,
        "reserved_1yr_hourly": 5.50,
        "reserved_3yr_hourly": 4.00,
        "memory_gb": 80,
        "flops_bf16_tflops": 989,
        "cloud": "GCP / AWS / Azure",
    },
    "NVIDIA L4": {
        "on_demand_hourly": 0.75,
        "reserved_1yr_hourly": 0.45,
        "reserved_3yr_hourly": 0.32,
        "memory_gb": 24,
        "flops_bf16_tflops": 121.5,
        "cloud": "GCP",
    },
    "NVIDIA V100 (16GB)": {
        "on_demand_hourly": 2.48,
        "reserved_1yr_hourly": 1.49,
        "reserved_3yr_hourly": 1.04,
        "memory_gb": 16,
        "flops_bf16_tflops": 112,
        "cloud": "GCP / AWS",
    },
    "TPU v4 (per chip)": {
        "on_demand_hourly": 3.22,
        "reserved_1yr_hourly": 1.93,
        "reserved_3yr_hourly": 1.35,
        "memory_gb": 32,
        "flops_bf16_tflops": 275,
        "cloud": "GCP",
    },
}

LLM_API_PRICING = {
    "GPT-4 Turbo": {
        "input_per_1m": 10.0,
        "output_per_1m": 30.0,
        "provider": "OpenAI",
        "context_window_k": 128,
        "quality_score": 9,
    },
    "GPT-4o": {
        "input_per_1m": 5.0,
        "output_per_1m": 15.0,
        "provider": "OpenAI",
        "context_window_k": 128,
        "quality_score": 9,
    },
    "GPT-3.5 Turbo": {
        "input_per_1m": 0.5,
        "output_per_1m": 1.5,
        "provider": "OpenAI",
        "context_window_k": 16,
        "quality_score": 7,
    },
    "Claude 3 Opus": {
        "input_per_1m": 15.0,
        "output_per_1m": 75.0,
        "provider": "Anthropic",
        "context_window_k": 200,
        "quality_score": 10,
    },
    "Claude 3 Sonnet": {
        "input_per_1m": 3.0,
        "output_per_1m": 15.0,
        "provider": "Anthropic",
        "context_window_k": 200,
        "quality_score": 8,
    },
    "Claude 3 Haiku": {
        "input_per_1m": 0.25,
        "output_per_1m": 1.25,
        "provider": "Anthropic",
        "context_window_k": 200,
        "quality_score": 7,
    },
    "Gemini 1.5 Pro": {
        "input_per_1m": 3.5,
        "output_per_1m": 10.5,
        "provider": "Google",
        "context_window_k": 1000,
        "quality_score": 8,
    },
    "Gemini 1.5 Flash": {
        "input_per_1m": 0.35,
        "output_per_1m": 1.05,
        "provider": "Google",
        "context_window_k": 1000,
        "quality_score": 7,
    },
}

OPEN_SOURCE_MODELS = {
    "Mistral 7B": {"params_b": 7, "gpu_memory_gb": 16, "tokens_per_sec_a100": 120},
    "LLaMA 3 8B": {"params_b": 8, "gpu_memory_gb": 16, "tokens_per_sec_a100": 100},
    "LLaMA 3 70B": {"params_b": 70, "gpu_memory_gb": 140, "tokens_per_sec_a100": 18},
    "Mixtral 8x7B": {"params_b": 47, "gpu_memory_gb": 96, "tokens_per_sec_a100": 40},
    "Falcon 40B": {"params_b": 40, "gpu_memory_gb": 80, "tokens_per_sec_a100": 22},
}

FINE_TUNING_GPU_HOURS_PER_BILLION_TOKENS = {
    "7B Model": 150,
    "13B Model": 280,
    "34B Model": 720,
    "70B Model": 1800,
}

CLOUD_STORAGE_PRICING = {
    "GCP Cloud Storage": 0.020,
    "AWS S3 Standard": 0.023,
    "Azure Blob Storage": 0.018,
}

CLOUD_NETWORK_EGRESS = {
    "GCP": 0.08,
    "AWS": 0.09,
    "Azure": 0.087,
}

VECTOR_DB_PRICING = {
    "Pinecone (Managed)": {"per_1m_vectors_month": 70, "query_per_1m": 8},
    "Weaviate Cloud": {"per_1m_vectors_month": 50, "query_per_1m": 5},
    "Qdrant Cloud": {"per_1m_vectors_month": 36, "query_per_1m": 4},
    "Self-hosted (compute)": {"per_1m_vectors_month": 20, "query_per_1m": 1.5},
}

LABOR_RATES = {
    "ML Engineer": 160000,
    "Data Engineer": 140000,
    "DevOps / MLOps Engineer": 145000,
    "Data Scientist": 150000,
    "Engineering Manager": 175000,
}

LEGACY_SYSTEM_BENCHMARKS = {
    "Human-only Support (per agent/yr)": 75000,
    "Rule-based Chatbot": 8000,
    "Traditional IVR System": 5000,
    "Outsourced BPO (per seat/yr)": 45000,
    "Knowledge Management Software": 12000,
}
