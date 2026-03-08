import sqlite3
import pandas as pd
from logger import get_logger

logger = get_logger(__name__)

DB_PATH = "data/consumerlens.db"

def run_query(sql):
    logger.info(f"Executing SQL query")
    conn = sqlite3.connect(DB_PATH)
    df = pd.read_sql_query(sql, conn)
    conn.close()
    logger.info(f"Rows returned: {len(df)}")
    return df
