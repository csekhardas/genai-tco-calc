import numpy as np


def compute_productivity_gains(
    num_employees_affected,
    hours_saved_per_employee_monthly,
    avg_hourly_rate,
    automation_rate_pct,
):
    direct_time_savings = num_employees_affected * hours_saved_per_employee_monthly * avg_hourly_rate
    automation_value = (automation_rate_pct / 100) * direct_time_savings * 0.5
    return {
        "direct_time_savings": direct_time_savings,
        "automation_value": automation_value,
        "total": direct_time_savings + automation_value,
    }


def compute_cost_avoidance(
    legacy_monthly_cost,
    headcount_reduction,
    avg_agent_annual_salary,
    error_reduction_pct,
    avg_error_cost,
    monthly_error_volume,
):
    headcount_savings = (headcount_reduction * avg_agent_annual_salary) / 12
    error_savings = (error_reduction_pct / 100) * monthly_error_volume * avg_error_cost
    legacy_savings = legacy_monthly_cost
    return {
        "legacy_replacement": legacy_savings,
        "headcount_reduction": headcount_savings,
        "error_reduction": error_savings,
        "total": legacy_savings + headcount_savings + error_savings,
    }


def compute_revenue_impact(
    monthly_customer_interactions,
    baseline_csat_score,
    target_csat_score,
    avg_revenue_per_customer,
    churn_reduction_pct,
    upsell_lift_pct,
):
    csat_improvement = (target_csat_score - baseline_csat_score) / 10
    churn_value = monthly_customer_interactions * (churn_reduction_pct / 100) * avg_revenue_per_customer
    upsell_value = monthly_customer_interactions * (upsell_lift_pct / 100) * avg_revenue_per_customer * 0.15
    sentiment_value = csat_improvement * monthly_customer_interactions * avg_revenue_per_customer * 0.02
    return {
        "churn_reduction": churn_value,
        "upsell_lift": upsell_value,
        "csat_improvement": sentiment_value,
        "total": churn_value + upsell_value + sentiment_value,
    }


def compute_payback_period(total_initial_investment, monthly_net_benefit):
    if monthly_net_benefit <= 0:
        return float("inf")
    return total_initial_investment / monthly_net_benefit


def compute_roi_over_time(
    monthly_tco,
    monthly_benefits,
    initial_investment,
    months=60,
    benefit_ramp_months=3,
    tco_growth_rate_pct=3,
    benefit_growth_rate_pct=8,
):
    records = []
    cumulative_cost = initial_investment
    cumulative_benefit = 0.0

    for m in range(1, months + 1):
        ramp_factor = min(1.0, m / benefit_ramp_months)
        monthly_cost = monthly_tco * ((1 + tco_growth_rate_pct / 100 / 12) ** (m - 1))
        monthly_benefit = monthly_benefits * ramp_factor * ((1 + benefit_growth_rate_pct / 100 / 12) ** (m - 1))
        cumulative_cost += monthly_cost
        cumulative_benefit += monthly_benefit
        net = cumulative_benefit - cumulative_cost
        roi_pct = ((cumulative_benefit - cumulative_cost) / cumulative_cost * 100) if cumulative_cost > 0 else 0

        records.append({
            "month": m,
            "monthly_cost": monthly_cost,
            "monthly_benefit": monthly_benefit,
            "cumulative_cost": cumulative_cost,
            "cumulative_benefit": cumulative_benefit,
            "net_value": net,
            "roi_pct": roi_pct,
        })

    return records


def compute_npv(monthly_net_cashflows, initial_investment, annual_discount_rate=0.10):
    monthly_rate = annual_discount_rate / 12
    npv = -initial_investment
    for i, cf in enumerate(monthly_net_cashflows, start=1):
        npv += cf / ((1 + monthly_rate) ** i)
    return npv


def compute_irr_approx(monthly_net_cashflows, initial_investment):
    cashflows = [-initial_investment] + monthly_net_cashflows
    # Bisection method for monthly IRR
    lo, hi = -0.5, 10.0
    for _ in range(100):
        mid = (lo + hi) / 2
        npv = sum(cf / ((1 + mid) ** i) for i, cf in enumerate(cashflows))
        if npv > 0:
            lo = mid
        else:
            hi = mid
        if abs(hi - lo) < 1e-7:
            break
    monthly_irr = (lo + hi) / 2
    annual_irr = (1 + monthly_irr) ** 12 - 1
    return annual_irr * 100


def compute_full_roi(
    monthly_tco,
    productivity_gains,
    cost_avoidance,
    revenue_impact,
    initial_investment,
    months=36,
):
    monthly_benefits = productivity_gains["total"] + cost_avoidance["total"] + revenue_impact["total"]
    timeline = compute_roi_over_time(monthly_tco, monthly_benefits, initial_investment, months)
    payback_months = compute_payback_period(initial_investment, monthly_benefits - monthly_tco)
    net_cashflows = [r["monthly_benefit"] - r["monthly_cost"] for r in timeline]
    npv = compute_npv(net_cashflows, initial_investment)
    try:
        irr = compute_irr_approx(net_cashflows, initial_investment)
    except Exception:
        irr = None

    return {
        "monthly_benefits": monthly_benefits,
        "monthly_net": monthly_benefits - monthly_tco,
        "payback_months": payback_months,
        "npv": npv,
        "irr_pct": irr,
        "timeline": timeline,
        "benefit_breakdown": {
            "Productivity Gains": productivity_gains["total"],
            "Cost Avoidance": cost_avoidance["total"],
            "Revenue Impact": revenue_impact["total"],
        },
    }
