# ConsumerLens

## Overview

**ConsumerLens** is a Natural Language Consumer Intelligence Engine that enables users to explore large consumer review datasets using simple English questions.

Instead of writing SQL queries or manually performing exploratory data analysis, users can ask questions such as:

- *“Show rating trend in the US.”*
- *“Compare ratings between US and GB.”*
- *“Which country has the highest average rating?”*

**ConsumerLens** translates these natural language questions into structured query plans, generates SQL queries safely, retrieves aggregated insights from the database, and produces concise analytical observations.

The application exposes these capabilities through a REST API.

### Intent

The primary intent of ConsumerLens is to demonstrate how **Large Language Models (LLMs)** can be used to bridge the gap between **natural language queries and structured data analytics**.  

It aims to:

- Enable intuitive interaction with consumer data
- Remove the dependency on manual SQL-based exploration
- Provide explainable insights derived from structured datasets
- Demonstrate how AI can power consumer intelligence platforms

### Objective

The core objectives of this application are:

1. **Natural Language Querying**

   Allow users to query consumer review datasets using plain English.

2. **Structured Query Planning**

   Convert user questions into structured query plans using an LLM while maintaining strict guardrails.

3. **Safe SQL Generation**

   Generate deterministic SQL queries using validated query plans.

4. **Insight Generation**

   Use LLMs to summarize aggregated results and generate meaningful consumer insights.

5. **API-Based Architecture**

   Provide a clean REST API that can be easily consumed by dashboards, analytics tools, or web interfaces.

### Architecture

ConsumerLens follows a modular pipeline architecture:

```sh
User Question
    ↓
Planner (LLM)
    ↓
Query Plan Validation
    ↓
SQL Builder
    ↓
Database Execution
    ↓
Insight Generator (LLM)
    ↓
API Response
```

### Components

| Component | Responsibility |
| --------- | ------------- |
| **Planner** | Converts natural language queries into structured query plans |
| **Validators** | Ensures query plans conform to supported analytics operations |
| **SQL Builder** | Deterministically constructs SQL queries from query plans |
| **Executor** | Executes SQL queries against the SQLite database |
| **Insight Generator** | Produces human-readable insights from aggregated results |
| **API Layer** | Exposes the analytics pipeline via REST endpoints |

#### Supported Query Intents

**ConsumerLens** currently supports the following analytics intents:

- `average_rating_by_country`
- `rating_trend_over_time`
- `review_volume_trend`
- `compare_countries`
- `low_rating_analysis`
- `filter_by_rating_threshold`

These intents allow structured exploration of consumer satisfaction trends and regional performance.

#### Dataset Description

The application operates on a dataset of consumer reviews containing approximately **20,000 records**.

| Column | Description |
| ------ | ----------- |
| `review_id` | Unique identifier for each review |
| `country` | Country where the review originated |
| `rating` | Consumer rating (1–5 scale) |
| `review_title` | Short title summarizing the review |
| `review_text` | Full consumer review content |
| `experience_date` | Date of the consumer experience |
| `year_month` | Derived field used for monthly trend analysis |

##### Key Characteristics

- Reviews span multiple years
- Multiple countries are represented
- Ratings range from **1 to 5**
- Suitable for trend analysis and sentiment exploration

##### Data Storage

The dataset is stored in a **SQLite database**:
> data/consumerlens.db

### API Endpoints

#### Health Check

> GET /health

```json
Response:
```json
{
  "status": "ok",
  "service": "consumerlens"
}
```

#### Analyze Consumer Data

> POST /analyze

```json
Request Body:
{
  "question": "Show rating trend in the US"
}
```

```json
Response:
{
  "query_plan": {...},
  "sql": "...",
  "data": [...],
  "insights": "...",
  "metadata": {
    "rows_returned": 42,
    "execution_time_ms": 120
  }
}
```

### Project Structure

```sh
consumerlens/
│
├── api.py
├── planner.py
├── sql_builder.py
├── executor.py
├── insights.py
├── validators.py
├── logger.py
├── config.py
│
├── prompts/
│   └── planner_prompt.txt
│
├── data/
│   └── consumerlens.db
│
├── .env
├── .gitignore
└── requirements.txt
```

#### Check into the project

> cd consumerlens  

#### Set the Environment

> python -m venv venv

#### Install the dependencies

> python -m venv venv

#### Configure the API_KEY

Create a .env file in the project root.

> OPENAI_API_KEY=<openai_api_key>

#### Running the Application

> uvicorn api:app --reload

#### Interactive API Documentation

> <http://localhost:8000/docs>

## Conclusion

**ConsumerLens** demonstrates how natural language interfaces combined with structured analytics pipelines can enable powerful consumer intelligence systems.  

The project highlights how LLMs can be integrated with traditional data systems to provide accessible and explainable insights for consumer analytics.  

---

### Example Questions for ConsumerLens

Below are example natural language questions that can be asked using the ConsumerLens API.  

### Rating Trend Over Time

- Show rating trend in the `US`
- Show rating trend in `GB`
- Show rating trend in the `US` and `GB`
- How have ratings changed over time in the `US`
- Show average rating trend in `Canada`

### Average Rating by Country

- Show average rating by country
- Which country has the highest average rating
- Compare average rating across countries
- What is the average rating for each country
- Rank countries by average rating

### Review Volume Trend

- Show review volume trend over time
- How many reviews were posted each month
- Show monthly review counts
- How has review volume changed over time
- Show review count trend in the `US`

### Compare Countries

- Compare `US` and `GB` ratings
- Compare ratings between `US` and `Canada`
- Compare review trends for `US` and `GB`
- Which country has better ratings, `US` or `GB`
- Compare average ratings in `US` and `India`

### Low Rating Analysis

- Which country has the lowest ratings
- Show countries with lowest average rating
- Which markets have poor ratings
- Where are ratings lowest
- Which country has the worst ratings

### Rating Threshold Filters

- Show reviews with `rating <= 2` in the `US`
- Count reviews with `rating <= 1` in `GB`
- Show number of low rating reviews in `Canada`
- How many reviews have `rating <= 2`
- Show low rating review count by country

---
