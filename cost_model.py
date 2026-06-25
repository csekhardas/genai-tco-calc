import numpy as np
from .pricing_data import GPU_PRICING, LLM_API_PRICING, CLOUD_STORAGE_PRICING, CLOUD_NETWORK_EGRESS


def compute_api_model_cost(model_name, monthly_queries, avg_input_tokens, avg_output_tokens):
    if model_name not in LLM_API_PRICING:
        return 0.0
    p = LLM_API_PRICING[model_name]
    input_cost = (monthly_queries * avg_input_tokens / 1_000_000) * p["input_per_1m"]
    output_cost = (monthly_queries * avg_output_tokens / 1_000_000) * p["output_per_1m"]
    return input_cost + output_cost


def compute_gpu_monthly_cost(gpu_type, gpu_count, utilization_pct, pricing_tier="on_demand"):
    if gpu_type not in GPU_PRICING:
        return 0.0
    g = GPU_PRICING[gpu_type]
    rate_key = {
        "on_demand": "on_demand_hourly",
        "reserved_1yr": "reserved_1yr_hourly",
        "reserved_3yr": "reserved_3yr_hourly",
    }.get(pricing_tier, "on_demand_hourly")
    hours_per_month = 730
    active_hours = hours_per_month * (utilization_pct / 100)
    idle_hours = hours_per_month - active_hours
    idle_rate = g[rate_key] * 0.15  # idle instances still cost ~15% for reserved capacity
    cost = gpu_count * (active_hours * g[rate_key] + idle_hours * idle_rate)
    return cost


def compute_infrastructure_cost(
    deployment_type,
    gpu_type,
    gpu_count,
    utilization_pct,
    pricing_tier,
    storage_gb,
    cloud_provider,
    networking_egress_gb,
):
    gpu_cost = 0.0
    if deployment_type in ("Self-hosted (On-Prem)", "Cloud VM (Self-managed)"):
        gpu_cost = compute_gpu_monthly_cost(gpu_type, gpu_count, utilization_pct, pricing_tier)

    storage_rate = CLOUD_STORAGE_PRICING.get(f"{cloud_provider} Cloud Storage", 0.020)
    storage_cost = storage_gb * storage_rate

    network_rate = CLOUD_NETWORK_EGRESS.get(cloud_provider, 0.085)
    network_cost = networking_egress_gb * network_rate

    # Baseline cloud overhead: load balancer, monitoring, logging
    cloud_overhead = 150 if deployment_type != "Self-hosted (On-Prem)" else 50

    return {
        "gpu_compute": gpu_cost,
        "storage": storage_cost,
        "networking": network_cost,
        "cloud_overhead": cloud_overhead,
        "total": gpu_cost + storage_cost + network_cost + cloud_overhead,
    }


def compute_data_pipeline_cost(
    monthly_data_ingestion_gb,
    transformation_compute_hours,
    vector_db_vectors_m,
    monthly_vector_queries_m,
    vector_db_type,
    cloud_provider,
):
    from .pricing_data import VECTOR_DB_PRICING

    ingestion_cost = monthly_data_ingestion_gb * 0.10  # ~$0.10/GB ingest + ETL
    transform_rate = 0.384  # standard 8-core VM per hour
    transform_cost = transformation_compute_hours * transform_rate

    vdb = VECTOR_DB_PRICING.get(vector_db_type, {"per_1m_vectors_month": 50, "query_per_1m": 5})
    vector_storage_cost = vector_db_vectors_m * vdb["per_1m_vectors_month"]
    vector_query_cost = monthly_vector_queries_m * vdb["query_per_1m"]

    storage_rate = CLOUD_STORAGE_PRICING.get(f"{cloud_provider} Cloud Storage", 0.020)
    raw_storage_cost = monthly_data_ingestion_gb * 3 * storage_rate  # 3x for raw + processed + backup

    total = ingestion_cost + transform_cost + vector_storage_cost + vector_query_cost + raw_storage_cost
    return {
        "ingestion": ingestion_cost,
        "transformation": transform_cost,
        "vector_db_storage": vector_storage_cost,
        "vector_db_queries": vector_query_cost,
        "raw_storage": raw_storage_cost,
        "total": total,
    }


def compute_labor_cost(team_composition, allocation_pct):
    from .pricing_data import LABOR_RATES
    total_monthly = 0.0
    breakdown = {}
    for role, headcount in team_composition.items():
        annual = LABOR_RATES.get(role, 140000)
        monthly = annual / 12
        allocated = monthly * headcount * (allocation_pct / 100)
        breakdown[role] = allocated
        total_monthly += allocated
    return {"breakdown": breakdown, "total": total_monthly}


def compute_maintenance_cost(infra_total, pipeline_total, maintenance_pct=15):
    return (infra_total + pipeline_total) * (maintenance_pct / 100)


def compute_fine_tuning_cost(model_size, training_tokens_b, gpu_type, pricing_tier):
    from .pricing_data import FINE_TUNING_GPU_HOURS_PER_BILLION_TOKENS
    gpu_hours_per_b = FINE_TUNING_GPU_HOURS_PER_BILLION_TOKENS.get(model_size, 280)
    total_gpu_hours = gpu_hours_per_b * training_tokens_b
    if gpu_type not in GPU_PRICING:
        return 0.0
    rate_key = {
        "on_demand": "on_demand_hourly",
        "reserved_1yr": "reserved_1yr_hourly",
        "reserved_3yr": "reserved_3yr_hourly",
    }.get(pricing_tier, "on_demand_hourly")
    return total_gpu_hours * GPU_PRICING[gpu_type][rate_key]


def compute_total_tco(infra, model_api, data_pipeline, labor, maintenance, months=12):
    monthly = infra["total"] + model_api + data_pipeline["total"] + labor["total"] + maintenance
    return {
        "monthly": monthly,
        "annual": monthly * months,
        "three_year": monthly * 36,
        "five_year": monthly * 60,
        "breakdown": {
            "Infrastructure": infra["total"],
            "Model / API": model_api,
            "Data Pipeline": data_pipeline["total"],
            "Labor": labor["total"],
            "Maintenance": maintenance,
        },
    }


def compute_scenario_comparison(base_inputs):
    scenarios = {}

    # Scenario A: Cloud API (no GPU infra)
    api_cost = compute_api_model_cost(
        "GPT-4o",
        base_inputs["monthly_queries"],
        base_inputs["avg_input_tokens"],
        base_inputs["avg_output_tokens"],
    )
    api_infra = {
        "gpu_compute": 0,
        "storage": base_inputs["storage_gb"] * 0.020,
        "networking": base_inputs["networking_egress_gb"] * 0.08,
        "cloud_overhead": 150,
        "total": base_inputs["storage_gb"] * 0.020 + base_inputs["networking_egress_gb"] * 0.08 + 150,
    }
    api_total = api_cost + api_infra["total"]
    scenarios["Cloud API (GPT-4o)"] = {"total": api_total, "model": api_cost, "infra": api_infra["total"]}

    # Scenario B: Fine-tuned open-source on cloud GPU
    ft_inference_gpu_cost = compute_gpu_monthly_cost(
        "NVIDIA A100 (80GB)", 2, base_inputs["utilization_pct"], "reserved_1yr"
    )
    scenarios["Fine-tuned (LLaMA 70B, A100 x2)"] = {
        "total": ft_inference_gpu_cost + api_infra["total"],
        "model": ft_inference_gpu_cost,
        "infra": api_infra["total"],
    }

    # Scenario C: Self-hosted on-prem
    onprem_gpu = compute_gpu_monthly_cost(
        "NVIDIA A100 (80GB)", 4, base_inputs["utilization_pct"], "on_demand"
    )
    onprem_infra = onprem_gpu + base_inputs["storage_gb"] * 0.005 + 200  # on-prem storage cheaper
    scenarios["Self-hosted On-Prem (A100 x4)"] = {
        "total": onprem_infra,
        "model": onprem_gpu,
        "infra": base_inputs["storage_gb"] * 0.005 + 200,
    }

    # Scenario D: Cheaper API (Claude Haiku)
    haiku_cost = compute_api_model_cost(
        "Claude 3 Haiku",
        base_inputs["monthly_queries"],
        base_inputs["avg_input_tokens"],
        base_inputs["avg_output_tokens"],
    )
    scenarios["Cloud API (Claude Haiku)"] = {
        "total": haiku_cost + api_infra["total"],
        "model": haiku_cost,
        "infra": api_infra["total"],
    }

    return scenarios


def project_costs_over_time(monthly_tco, months=36, growth_rate_pct=5):
    costs = []
    cumulative = 0
    for m in range(1, months + 1):
        monthly = monthly_tco * ((1 + growth_rate_pct / 100 / 12) ** (m - 1))
        cumulative += monthly
        costs.append({"month": m, "monthly": monthly, "cumulative": cumulative})
    return costs
