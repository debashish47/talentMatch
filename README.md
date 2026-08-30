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

## RAG job assistant setup

The candidate **Job assistant** uses the same RAG pattern as the reference project: it indexes live open job posts in ChromaDB, retrieves and reranks relevant roles, asks Gemini to produce a grounded answer, and caches semantically similar questions in memory.

Create `backend/.env` from [backend/.env.example](backend/.env.example) and add your Gemini key:

```env
GEMINI_API_KEY=your_google_gemini_api_key
GEMINI_MODEL=gemini-2.0-flash
```

Install the refreshed backend requirements and restart the API:

```bash
cd backend
pip install -r requirements.txt
uvicorn app.main:app --reload
```

Chroma’s local job index is generated under `backend/chroma_db/` and is deliberately excluded from Git.

## Design notes

AI Match embeds each open job when an admin creates or edits it, stores that vector in ChromaDB, then embeds a candidate request and returns the nearest live jobs by cosine similarity. The results show the matched required skills and support applying directly. A small keyword/alias extractor remains only to make matched requirements readable. SQLite auto-creates and seeds three sample jobs on first run.

## API surface

- `/api/profiles` — create, read, update candidate profiles
- `/api/jobs` — browse/filter and admin CRUD job listings
- `/api/applications` — apply, view applications, and update pipeline state
- `/api/admin/dashboard` — applications per job, skill distribution, pipeline counts
- `/api/ai/match-jobs` — preference extraction, ranked matches, and explanations
- `/api/chat` — candidate RAG job assistant with grounding, reranking, and semantic cache
