import io
from contextlib import asynccontextmanager
from fastapi import FastAPI, File, Form, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pypdf
import uvicorn

from rag_engine import RAGSystem

# 1. Initialize RAG
rag_system = RAGSystem()


@asynccontextmanager
async def lifespan(app: FastAPI):
    rag_system.initialize()
    yield


# 2. Create FastAPI app
app = FastAPI(title="RAG Backend API", lifespan=lifespan)

# 3. Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Request schemas
class ChatRequest(BaseModel):
    message: str


class RoadmapRequest(BaseModel):
    company: str
    role: str


class ExperienceRequest(BaseModel):
    company: str


# 4. Extract text from uploaded file
async def extract_text(file: UploadFile) -> str:
    try:
        content = await file.read()
        filename = (file.filename or "").lower()

        if filename.endswith(".pdf"):
            pdf = pypdf.PdfReader(io.BytesIO(content))
            text = ""
            for page in pdf.pages:
                text += page.extract_text() or ""
            return text.strip()

        return content.decode("utf-8", errors="ignore").strip()
    except Exception as e:
        print(f"Error extracting text from {file.filename}: {e}")
        return ""


@app.get("/")
async def root():
    return {"status": "ok", "message": "RAG Backend Server is running"}


# 5. Chat
@app.post("/chat")
async def chat(request: ChatRequest):
    response = rag_system.chat(request.message)
    return {"response": response}


# 6. Generate roadmap
@app.post("/roadmap")
async def roadmap(request: RoadmapRequest):
    return rag_system.generate_roadmap(request.company, request.role)


# 7. Analyze skills
@app.post("/analyze-skills")
async def analyze_skills(
    company: str = Form(...),
    role: str = Form(...),
    file: UploadFile = File(...),
):
    text = await extract_text(file)
    if not text:
        return {"error": "Could not extract text from file"}
    return rag_system.analyze_skills(company, role, text)


# 8. Analyze ATS
@app.post("/analyze-ats")
async def analyze_ats(
    company: str = Form(...),
    role: str = Form(...),
    file: UploadFile = File(...),
):
    text = await extract_text(file)
    if not text:
        return {"error": "Could not extract text from file"}
    return rag_system.analyze_resume_ats(company, role, text)


# 9. Get company experiences
@app.post("/experiences")
async def experiences(request: ExperienceRequest):
    return rag_system.get_experiences(request.company)


# 10. Run server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)