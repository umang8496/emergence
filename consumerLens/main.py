# from planner import generate_plan
# from sql_builder import build_sql
# from executor import run_query
# from insights import generate_insights
# from logger import get_logger

# logger = get_logger(__name__)
# question = "Show rating trend in the US and GB"
# logger.info("ConsumerLens pipeline started")

# system_prompt = open("prompts/planner_prompt.txt").read()
# plan = generate_plan(question, system_prompt)
# sql = build_sql(plan)
# df = run_query(sql)
# insight = generate_insights(df)
# logger.info("Pipeline completed successfully")

# print("\nSQL:", sql)
# print("\nData:", df.head())
# print("\nInsights:", insight)
