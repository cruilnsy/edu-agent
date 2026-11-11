# backend/api/router.py
# All API routers, mounted under /api/v1 in main.py.

from fastapi import APIRouter

from backend.api.v1 import auth, exam, interview, qa, resume, unified_chat

api_router = APIRouter()

api_router.include_router(auth.router,          prefix="/auth",      tags=["Auth"])
api_router.include_router(unified_chat.router,  prefix="/chat",      tags=["Assistant"])
api_router.include_router(qa.router,            prefix="/qa",        tags=["Q&A"])
api_router.include_router(exam.router,          prefix="/exam",      tags=["Exam"])
api_router.include_router(resume.router,        prefix="/resume",    tags=["Resume"])
api_router.include_router(interview.router,     prefix="/interview", tags=["Interview"])
