ALLOWED_INTENTS = {
    "average_rating_by_country",
    "rating_trend_over_time",
    "review_volume_trend",
    "compare_countries",
    "low_rating_analysis",
    "filter_by_rating_threshold"
}

ALLOWED_METRICS = {"average_rating", "review_count"}

ALLOWED_GROUPS = {"country", "year_month"}

def validate_plan(plan: dict):
    """
    Validate LLM generated query plan before SQL generation.
    """
    if "intent" not in plan:
        raise ValueError("Query plan missing intent")

    if plan["intent"] not in ALLOWED_INTENTS:
        raise ValueError(f"Unsupported intent: {plan['intent']}")

    if "metrics" in plan:
        for metric in plan["metrics"]:
            if metric not in ALLOWED_METRICS:
                raise ValueError(f"Unsupported metric: {metric}")

    if "group_by" in plan:
        for group in plan["group_by"]:
            if group not in ALLOWED_GROUPS:
                raise ValueError(f"Unsupported grouping: {group}")

    return True
