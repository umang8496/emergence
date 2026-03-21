# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Repository Overview

**Emergence** is a comprehensive educational monorepo covering Claude, LLMs, and AI agents. It contains multiple projects demonstrating different aspects of working with Claude: from prompt engineering to API integration to building AI-powered applications.

This is a **monorepo structure** where each subdirectory is a self-contained educational project with its own tech stack and patterns. Projects range from Jupyter notebooks to Python applications to React applications.

## Repository Structure

```sh
emergence/
├── README.md                                          # Main index linking to all projects
├── CLAUDE.md                                          # This file
│
├── ai-agents/                                         # Agent implementations
│   └── code-reviewer/                                 # Example AI agent
│
├── building-with-claude-api/                          # Jupyter notebooks on Claude API
│   └── accessing-claude-with-the-api/notebook.ipynb
│
├── claude-101/                                        # Educational content
│
├── claude-api-101/                                    # Claude API fundamentals
│   └── notebooks with examples
│
├── claude-code-in-action/                             # Claude Code course with sample app
│   ├── README.md                                      # Course notes (comprehensive guide to Claude Code)
│   ├── uigen/                                         # Full-stack React app (see uigen/CLAUDE.md)
│   │   ├── package.json
│   │   ├── CLAUDE.md                                  # Project-specific guidance
│   │   ├── src/
│   │   ├── next.config.js
│   │   └── prisma/
│   │
│   └── uigen/ (other content)
│
├── consumerLens/                                      # Python FastAPI consumer analytics engine
│   ├── requirements.txt
│   ├── api.py                                         # FastAPI application
│   ├── planner.py                                     # NL to query plan (LLM)
│   ├── sql_builder.py                                 # Query plan to SQL
│   ├── executor.py                                    # Execute SQL against SQLite
│   ├── insights.py                                    # Generate insights (LLM)
│   ├── data/consumerlens.db                           # SQLite database
│   └── prompts/
│
├── introduction-to-llm-agents/                        # LLM agents course
│   └── notebooks
│
└── prompt-engineering-and-llm-application-development/ # Prompt engineering course
    └── notebooks
```

## Project Categories

### 1. Full-Stack Applications

#### claude-code-in-action/uigen/

- **Type**: Next.js + React full-stack application
- **Purpose**: AI-powered component generator with real-time preview
- **Key Tech**: Next.js 15, React 19, Prisma, Tailwind, Radix UI, Claude API
- **Commands**: See `claude-code-in-action/uigen/CLAUDE.md` for detailed setup and commands
- **Entry**: `npm run dev` to start at http://localhost:3000

#### consumerLens/

- **Type**: Python FastAPI backend service
- **Purpose**: Natural language interface for consumer data analytics
- **Key Tech**: Python, FastAPI, SQLite, Claude API
- **Setup**:
  
  ```bash
  cd consumerLens
  python -m venv venv
  source venv/bin/activate
  pip install -r requirements.txt
  ANTHROPIC_API_KEY=your-key uvicorn api:app --reload
  ```

- **API Docs**: http://localhost:8000/docs
- **Architecture**: NL Question → Query Planner (LLM) → SQL Builder → Executor → Insights (LLM) → API Response

### 2. Educational Notebooks

#### claude-api-101/, building-with-claude-api/, introduction-to-llm-agents/, prompt-engineering-and-llm-application-development/

- **Type**: Jupyter notebooks
- **Purpose**: Hands-on learning materials for various Claude topics
- **How to Run**: Open `.ipynb` files directly in Jupyter or Claude Code's notebook viewer
- **No setup required** beyond having Jupyter installed

### 3. Reference Content

#### claude-101/, ai-agents/

- **Type**: Educational resources and code examples
- **Purpose**: Learning materials and agent implementations
- **No build/run step** — browse directly

## Common Development Commands

### For Node.js Projects (uigen)

```bash
cd claude-code-in-action/uigen

# Setup
npm run setup           # Install deps, generate Prisma, run migrations

# Development
npm run dev            # Start dev server with Turbopack
npm run dev:daemon     # Run in background
npm run build          # Build for production
npm start              # Run production server

# Code quality
npm run lint           # ESLint check
npm test               # Vitest tests
npm test -- <file>     # Run specific test file

# Database
npm run db:reset       # Reset database (destructive)
```

### For Python Projects (consumerLens)

```bash
cd consumerLens

# Setup
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Running
ANTHROPIC_API_KEY=your-key uvicorn api:app --reload

# Testing
# (No test framework configured — manually test via /docs endpoint)
```

### For Notebooks

- Open in Jupyter: `jupyter notebook <path-to-notebook>`
- Run in Claude Code: Click the "run" button in the notebook viewer

## High-Level Architecture Patterns

### Full-Stack App (uigen)

#### Virtual File System Architecture

- In-memory file system (no disk I/O) simulating a code editor
- Files persisted to database as JSON strings, not to disk
- All operations via `VirtualFileSystem` class

#### AI Tool Integration

- Claude generates React components via two tools: `str_replace_editor` and `file_manager`
- Tools operate on virtual file system, triggering UI refreshes
- Real-time preview via Babel JSX transformation

#### Authentication & Data

- JWT-based auth with HTTP-only cookies
- Prisma ORM manages Users, Projects, and Messages
- Project state (file system + chat history) stored as JSON in database

### Python Service (consumerLens)

#### Pipeline Architecture

- LLM Planner: NL question → Structured query plan
- Validator: Ensure plan conforms to allowed operations
- SQL Builder: Deterministically construct SQL from plan
- Executor: Run SQL against SQLite database
- Insight Generator: LLM summarizes results

#### Supported Intents

- `average_rating_by_country`
- `rating_trend_over_time`
- `review_volume_trend`
- `compare_countries`
- `low_rating_analysis`
- `filter_by_rating_threshold`

## Key Architectural Decisions

### Why This Structure?

1. **Monorepo Layout**: Each project is independent yet discoverable. Allows teaching different patterns (notebooks, full-stack, services) in one place.

2. **Virtual File System (uigen)**: Eliminates disk complexity in the tutorial app. Files live in memory + database, making state management explicit and testable.

3. **Python Service (consumerLens)**: Demonstrates LLM integration with traditional structured data, showing how to safely translate NL queries to SQL.

4. **Notebook-Based Learning**: Jupyter notebooks are ideal for interactive learning and exploring Claude API step-by-step.

## Important Notes for Development

### Environment Variables

- **uigen**: Optional `ANTHROPIC_API_KEY` — app works with mock provider if not set
- **consumerLens**: Required `ANTHROPIC_API_KEY` for actual LLM features

### Database Considerations

- **uigen**: Uses Prisma + SQLite. Run `npm run db:reset` to wipe and recreate (destructive).
- **consumerLens**: Static SQLite database at `data/consumerlens.db` — read-only for normal usage.

### Testing

- **uigen**: `npm test` runs Vitest with React Testing Library. Tests colocated with source.
- **consumerLens**: No automated tests — validate via API docs at `/docs`.

## Working with Multiple Projects

Since this is a monorepo:

1. **Change directory to the project you're working on** — each has independent dependencies and commands
2. **Do not install from the root** — `npm install` at root won't work. Install within each project directory.
3. **Each project is self-contained** — modifications in one project don't affect others unless explicitly linked.

If working on the full-stack app (uigen), refer to `claude-code-in-action/uigen/CLAUDE.md` for detailed component architecture and patterns.

## Project Interdependencies

- **No hard dependencies between projects** — each stands alone
- **Course notes in claude-code-in-action/README.md** reference and explain concepts used across all projects
- Notebooks in other directories complement the full-stack app as learning materials

## Quick Navigation

| Project | Type | Start Command | Purpose |
| ------- | ---- | ------------- | ------- |
| uigen | Full-stack App | `cd claude-code-in-action/uigen && npm run dev` | AI component generator with live preview |
| consumerLens | Python API | `cd consumerLens && uvicorn api:app --reload` | NL analytics on consumer data |
| Notebooks | Educational | Open in Jupyter | Learn Claude API, prompting, agents |
| Content | Reference | Browse directly | Educational materials |
