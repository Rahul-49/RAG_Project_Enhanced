import json
import os
import re
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

load_dotenv()

# Configuration
CHROMA_DB_DIR = os.path.join(os.getcwd(), "chroma_db")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
os.environ["HF_HOME"] = os.path.join(os.getcwd(), "hf_cache")


def _clean_json(text: str) -> str:
    """Strip markdown code fences and whitespace from LLM response."""
    text = text.strip()
    match = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", text)
    if match:
        return match.group(1).strip()
    if text.startswith("```json"):
        text = text[7:]
    elif text.startswith("```"):
        text = text[3:]
    if text.endswith("```"):
        text = text[:-3]
    return text.strip()


class RAGSystem:
    def __init__(self):
        self.vectorstore = None
        self.llm = None
        self.reranker = None
        self.embeddings = None
        self.is_initialized = False

    def initialize(self):
        print("Initializing RAG Engine (Inference Only)...")

        if not GROQ_API_KEY:
            print("CRITICAL WARNING: GROQ_API_KEY not found in env. Please add it to .env file.")

        # 1. Initialize Embeddings
        print("Loading Embeddings (sentence-transformers/all-mpnet-base-v2)...")
        self.embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

        # 2. Load Vector Store
        print("Loading existing Vector Store...")
        if not os.path.exists(CHROMA_DB_DIR):
            print("❌ ERROR: Vector DB not found at 'chroma_db/'. Please run 'python ingest.py' first.")
            self.vectorstore = None
        else:
            self.vectorstore = Chroma(persist_directory=CHROMA_DB_DIR, embedding_function=self.embeddings)
            print("Vector Store connected.")

        # 3. Reranker
        print("Loading Reranker (cross-encoder/ms-marco-MiniLM-L-6-v2)...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # 4. LLM
        print("Initializing LLM (Groq)...")
        self.llm = ChatGroq(
            temperature=0,
            model_name="openai/gpt-oss-120b",
            api_key=GROQ_API_KEY,
        )

        self.is_initialized = True
        print("✅ RAG Engine Ready for Queries.")

    def _get_context(self, query: str, k: int = 5) -> str:
        """Retrieve and format context string from vector store."""
        if not self.vectorstore:
            return ""
        try:
            docs = self.vectorstore.similarity_search(query, k=k)
            return "\n\n".join(doc.page_content for doc in docs)
        except Exception:
            return ""

    def _invoke_json(self, prompt: str, as_list: bool = False):
        """Invoke LLM and parse JSON response with error handling."""
        try:
            response = self.llm.invoke(prompt)
            return json.loads(_clean_json(response.content))
        except Exception as e:
            err = {"error": f"LLM/Parsing Error: {str(e)}"}
            return [err] if as_list else err

    def chat(self, user_query: str):
        if not self.is_initialized:
            return "System is initializing or failed. Check server logs."

        if not self.vectorstore:
            return "Knowledge base is empty. Please run the ingestion script."

        # 1. Retrieve
        try:
            retrieved_docs = self.vectorstore.similarity_search(user_query, k=10)
        except Exception as e:
            return f"Error accessing vector DB: {str(e)}"

        if not retrieved_docs:
            return "I couldn't find any relevant information in my knowledge base."

        # 2. Rerank
        pairs = [[user_query, doc.page_content] for doc in retrieved_docs]
        try:
            scores = self.reranker.predict(pairs)
            scored_docs = sorted(zip(retrieved_docs, scores), key=lambda x: x[1], reverse=True)
            top_docs = [doc for doc, _ in scored_docs[:3]]
        except Exception as e:
            print(f"Reranking error: {e}. Fallback to top 3 retrieval.")
            top_docs = retrieved_docs[:3]

        # 3. Generate
        context = "\n\n".join(doc.page_content for doc in top_docs)
        prompt = f"""You are a helpful assistant for PVPSIT College students.

Use the provided context as the ONLY source of factual information about
campus-related topics.

Context:
{context}

Rules:
1. If the user asks a factual question about information that could be
   answered from the context, answer using ONLY the provided context.
2. If the requested factual information is not present in the context,
   say exactly: "I'm unable to answer the question."
3. You may respond normally to conversational messages such as greetings,
   thanks, acknowledgements, confirmations, and casual conversation.
4. Do not invent campus-related facts that are not present in the context.
5. Keep responses helpful, concise, and natural.
"

Question: {user_query}

Answer:"""

        try:
            response = self.llm.invoke(prompt)
            return response.content
        except Exception as e:
            return f"Error communicating with LLM: {str(e)}"

    def generate_roadmap(self, company: str, role: str):
        if not self.is_initialized:
            return {"error": "System is initializing"}

        query = f"preparation roadmap for {company} {role} interview recruitment process"
        try:
            retrieved_docs = self.vectorstore.similarity_search(query, k=5)
        except Exception as e:
            return {"error": f"Vector DB Error: {str(e)}"}

        if not retrieved_docs:
            return {"error": "No info found"}

        context = "\n\n".join(doc.page_content for doc in retrieved_docs)
        prompt = f"""You are an expert career coach. Create a preparation roadmap for the role of '{role}' at '{company}' based strictly on the context provided.

Context:
{context}

Requirements:
- Return ONLY a valid JSON array of objects.
- Do not wrap in markdown code blocks (like ```json ... ```), just raw JSON.
- Each object must have these keys:
  - "title": string (Topic name, e.g., "Quantitative Aptitude")
  - "status": string (Set all to "pending")
  - "date": string (Estimated timeline, e.g., "Week 1", "Week 2")
  - "description": string (Brief actionable advice or specific topics to cover from context)

If the context lacks specific details, use general industry standards for that role/company but match the context's emphasis.
"""
        return self._invoke_json(prompt)

    def analyze_skills(self, company: str, role: str, resume_text: str):
        if not self.is_initialized:
            return {"error": "System is initializing"}

        context = self._get_context(f"technical skills requirements for {company} {role}", k=5)
        prompt = f"""You are an expert technical recruiter and career coach.

Role: {role}
Company: {company}

Job Requirements Context (from internal database):
{context}

Candidate's Resume Text:
{resume_text[:4000]}

Task:
Analyze the resume against the role requirements.

Requirements:
- Return ONLY a valid JSON object.
- Do not wrap in markdown code blocks.
- The JSON must have these exact keys:
  - "present_skills": ["Skill A", "Skill B", ...],
  - "missing_skills": ["Skill X", "Skill Y", ...],
  - "recommendations": [
      {{ "skill": "Skill X", "action": "Specific advice on how to learn/improve this." }},
      ...
    ]

CRITICAL VALIDATION STEP:
1. Check if "Candidate's Resume Text" is actually a resume or contains relevant professional details. 
2. If the text is garbage, empty, or completely unrelated (e.g., a random story, code snippet, or nonsensical text), return:
   {{ "present_skills": [], "missing_skills": ["Resume appears invalid or empty"], "recommendations": [] }}
3. ONLY extract skills that are EXPLICITLY mentioned in the resume text. Do NOT infer skills slightly related.
4. Be strict. If a skill is not there, it goes to "missing_skills".

Use the Context to identify specific tools/frameworks {company} prefers.
"""
        return self._invoke_json(prompt)

    def analyze_resume_ats(self, company: str, role: str, resume_text: str):
        if not self.is_initialized:
            return {"error": "System is initializing"}

        context = self._get_context(f"job description requirements keywords for {company} {role}", k=5)
        prompt = f"""You are an Advanced Applicant Tracking System (ATS) Analyzer and Resume Optimization Engine.

Target Role: {role}
Target Company: {company}

Official Job Description (Authoritative Context):
{context}

Candidate Resume (Input Text):
{resume_text[:4000]}

Objective:
Perform a deep ATS compatibility analysis of the candidate’s resume against the job description and generate actionable, role-specific improvements that maximize ATS ranking and recruiter relevance.

Instructions:
- Treat the job description as the single source of truth.
- Use ATS parsing logic, keyword weighting, semantic relevance, and role alignment.
- Do not invent experience; only optimize wording and structure.

CRITICAL VALIDATION STEP:
1. Analyze "Candidate Resume" text.
2. If the document is NOT a resume (e.g. random text, assignment, invalid characters), set "ats_score" to 0 and listing "Invalid Document" in "formatting_issues".
3. Do NOT hallucinate matches. Only count a keyword if it appears in the text.

Strict Output Requirements:
- Return ONLY valid JSON.
- Do NOT include explanations, markdown, or commentary.
- Suggestions must be direct, explicit resume modifications (ready to paste), not examples.

Required JSON Schema:
{{
  "ats_score": 0,
  "missing_keywords": [],
  "formatting_issues": [],
  "tailored_suggestions": []
}}

Field-Level Expectations:
- ats_score: Integer 0–100 reflecting true ATS match quality.
- missing_keywords: High-impact, role-specific keywords required by the job description.
- formatting_issues: ATS-parsing problems (tables, columns, headers, dates, titles).
- tailored_suggestions: Direct instructions with exact wording and placement
  (summary, experience, skills, projects). No soft language.

Critical Constraints:
- No hypothetical phrasing.
- Use authoritative, ATS-optimized language.
- Optimize for automated screening and recruiter review.
"""
        return self._invoke_json(prompt)

    def get_experiences(self, company: str):
        if not self.is_initialized:
            return {"error": "System is initializing"}

        context = self._get_context(f"interview experience {company} questions rounds", k=10)
        if not context:
            return []

        prompt = f"""You are a curator of interview experiences. Extract and summarize distinct interview experiences for '{company}' from the following context.

Context:
{context}

Requirements:
- Return ONLY a valid JSON array of objects.
- No markdown formatting.
- Each object must have:
  - "candidate_profile": string (e.g., "Fresher, CSE")
  - "role": string
  - "rounds": array of strings (List of rounds)
  - "questions_asked": array of strings (Key technical/HR questions)
  - "verdict": string ("Selected" or "Rejected")
  - "tips": string (Advice)

Ensure the data is extracted strictly from context where possible. If context is generic, synthesize a representative experience based on standard {company} patterns found in the text.
"""
        return self._invoke_json(prompt, as_list=True)
