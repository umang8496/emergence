from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import time

from planner import generate_plan
from sql_builder import build_sql
from executor import run_query
from insights import generate_insights
from validators import validate_plan

from logger import get_logger

logger = get_logger(__name__)

app = FastAPI(title="ConsumerLens API")

class QueryRequest(BaseModel):
    question: str

# ---------------------------
# Health Check Endpoint
# ---------------------------
@app.get("/health")
def health_check():
    return {
        "status": "ok",
        "service": "consumerlens"
    }


# ---------------------------
# Main Analytics Endpoint
# ---------------------------
@app.post("/analyze")
def analyze_query(request: QueryRequest):
    start_time = time.time()
    try:
        logger.info(f"Received question: {request.question}")
        system_prompt = open("prompts/planner_prompt.txt").read()

        # Step 1 — Planner
        plan = generate_plan(request.question, system_prompt)
        # Step 2 — Validation
        validate_plan(plan)
        # Step 3 — SQL Builder
        sql = build_sql(plan)
        # Step 4 — Execute SQL
        df = run_query(sql)
        # Step 5 — Insights
        insights = generate_insights(df)
        execution_time = int((time.time() - start_time) * 1000)
        response = {
            "query_plan": plan,
            "sql": sql,
            "data": df.to_dict(orient="records"),
            "insights": insights,
            "metadata": {
                "rows_returned": len(df),
                "execution_time_ms": execution_time
            }
        }
        return response
    except ValueError as e:
        logger.error(str(e))
        raise HTTPException(
            status_code=400,
            detail=str(e)
        )
    except Exception as e:
        logger.exception("Unexpected error")
        raise HTTPException(
            status_code=500,
            detail="Internal server error"
        )
