# Agentic Claims Pipeline (Google ADK + Streamlit + Langfuse)

This app implements an agentic insurance claim summarisation workflow built with:

- Google Agent Development Kit (ADK)
- Streamlit UI
- Langfuse observability

Features:

- Multi-step claim summarisation pipeline (SequentialAgent)
- Upload & parse claim PDFs / DOCX / TXT
- Q&A over the uploaded claim document
- Feedback-based summary regeneration
- Optional Langfuse tracing for pipeline, Q&A and regeneration

See `AGENTS_AND_EXECUTION.md` for a detailed description of agents and how to run the app
locally, via Docker, or on AWS EKS.
