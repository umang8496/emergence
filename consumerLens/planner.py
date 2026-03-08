import json
from config import client, OPENAI_MODEL
from logger import get_logger

logger = get_logger(__name__)

def generate_plan(question, system_prompt):
    logger.info(f"Received question: {question}")
    response = client.chat.completions.create(
        model=OPENAI_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": question}
        ],
        temperature=0
    )
    output = response.choices[0].message.content
    logger.info(f"Planner raw output: {output}")
    plan = json.loads(output)
    logger.info(f"Parsed plan: {plan}")
    return plan
