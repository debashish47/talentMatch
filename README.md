# TalentMatch — AI Job Board

A full-stack hiring-board MVP built with FastAPI, SQLite, React, and Vite. It has candidate profiles, job CRUD, filtering, applications and pipeline management, analytics, plus transparent natural-language job matching.

## Run it

Terminal 1:

```bash
cd backend
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Terminal 2:

```bash
cd frontend
npm install
npm run dev
```

Open `http://localhost:5173`. The backend API documentation is available at `http://localhost:8000/docs`.

## Design notes

Natural-language matching uses a safe local keyword/alias extractor, then deterministic scoring. Open roles alone are eligible. Scores include only preferences actually found in the query: skills (50), location (20), role (15), experience (10), and domain (5). This keeps results explainable and works without an API key. SQLite auto-creates and seeds three sample jobs on first run.

## API surface

- `/api/profiles` — create, read, update candidate profiles
- `/api/jobs` — browse/filter and admin CRUD job listings
- `/api/applications` — apply, view applications, and update pipeline state
- `/api/admin/dashboard` — applications per job, skill distribution, pipeline counts
- `/api/ai/match-jobs` — preference extraction, ranked matches, and explanations
