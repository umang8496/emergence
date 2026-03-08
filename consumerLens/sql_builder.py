from logger import get_logger
logger = get_logger(__name__)

def build_sql(plan):
    logger.info(f"Building SQL from plan: {plan}")
    base_query = "SELECT "

    if "average_rating" in plan["metrics"]:
        metric_part = "AVG(rating) as average_rating"
    elif "review_count" in plan["metrics"]:
        metric_part = "COUNT(*) as review_count"
    else:
        metric_part = "*"

    group_part = ""

    if plan["group_by"]:
        group_field = plan["group_by"][0]
        base_query += f"{group_field}, {metric_part} "
        group_part = f" GROUP BY {group_field} ORDER BY {group_field}"
    else:
        base_query += metric_part

    base_query += "FROM reviews"
    where_clauses = []

    if plan["filters"]["country"]:
        country_filter = plan["filters"]["country"]
        if isinstance(country_filter, list):
            countries = ",".join([f"'{c}'" for c in country_filter])
            where_clauses.append(f"country IN ({countries})")
        else:
            where_clauses.append(f"country = '{country_filter}'")

    if where_clauses:
        base_query += " WHERE " + " AND ".join(where_clauses)

    final_query = base_query + group_part + ";"
    logger.info(f"Generated SQL: {final_query}")
    return final_query
