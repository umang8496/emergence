from config import client, OPENAI_MODEL
from logger import get_logger

logger = get_logger(__name__)

def generate_insights(df):
    logger.info("Generating insights from aggregated data")
    prompt = f"""
        You are a consumer data analyst.

        The column `year_month` represents time in the format YYYY-MM
        (e.g., 2019-12 means December 2019).

        Each row represents the number of reviews posted in that month.

        Here is the aggregated data:

        {df.to_string(index=False)}

        Provide 2–3 insights about trends over time.
        Focus on long-term trends rather than individual months.
    """

    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": "You analyze consumer trends."},
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    insights = response.choices[0].message.content
    logger.info("Insights generated successfully")
    return insights
