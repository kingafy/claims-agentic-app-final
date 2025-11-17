import os

from google.adk.agents import LlmAgent, SequentialAgent

DEFAULT_MODEL = os.getenv("ADK_MODEL_NAME", "gemini-2.0-flash")


data_ingestion_agent = LlmAgent(
    name="DataIngestionAgent",
    model=DEFAULT_MODEL,
    description="Cleans raw claim text into normalized English paragraphs.",
    instruction=(
        "You are the first step in an insurance claims pipeline.\n"
        "Your input is the raw claim text stored in state under 'claim_text'.\n"
        "The content of {claim_text} may contain OCR noise, duplicates, and "
        "unnecessary headers/footers.\n\n"
        "Task:\n"
        "1. Normalize spacing, punctuation, and paragraph breaks.\n"
        "2. Remove page headers/footers and boilerplate that isn't relevant to "
        "understanding the incident.\n"
        "3. Preserve all factual details about the incident, damages, policy "
        "information, and recommendations.\n\n"
        "Return only the cleaned claim narrative as plain text."
    ),
    output_key="clean_claim_text",
)


parser_agent = LlmAgent(
    name="ParserAgent",
    model=DEFAULT_MODEL,
    description="Extracts key structured fields from the cleaned claim text.",
    instruction=(
        "You are a claims parser. You are given cleaned claim text below:\n\n"
        "{clean_claim_text}\n\n"
        "Extract key fields as JSON with the following keys:\n"
        "  - claimant_name (string or null)\n"
        "  - policy_number (string or null)\n"
        "  - date_of_loss (string, ISO if available, else as written, or null)\n"
        "  - location_of_loss (string or null)\n"
        "  - cause_of_loss (string)\n"
        "  - claimed_items (string - bullet style text is fine)\n"
        "  - coverage_issues (string - any potential coverage concerns or 'none')\n"
        "  - recommended_actions (string - short bullet list of next steps)\n\n"
        "Important:\n"
        "- If information is missing, use null for that field.\n"
        "- Output MUST be valid JSON only, with no backticks, no code fences, "
        "and no extra commentary."
    ),
    output_key="parsed_claim_json",
)


summary_agent = LlmAgent(
    name="SummaryAgent",
    model=DEFAULT_MODEL,
    description="Drafts a narrative claim summary from structured fields.",
    instruction=(
        "You are a claim summary writer. You are given the parsed claim as JSON:\n\n"
        "{parsed_claim_json}\n\n"
        "Write a concise, professional narrative summary suitable for a claims file.\n"
        "Guidelines:\n"
        "- 3–6 paragraphs.\n"
        "- Use neutral, objective language.\n"
        "- Cover background, incident details, investigation findings, liability, "
        "coverage, and next steps.\n"
        "- Do not restate obvious field labels; write as continuous prose."
    ),
    output_key="summary_draft",
)


formatter_agent = LlmAgent(
    name="FormatterAgent",
    model=DEFAULT_MODEL,
    description="Formats the draft summary into standardized sections.",
    instruction=(
        "You are a formatter in the claims pipeline. Starting from the draft "
        "summary below, reformat into clear sections:\n\n"
        "{summary_draft}\n\n"
        "Use Markdown with the exact headings:\n"
        "1. Incident Details\n"
        "2. Damage Assessment\n"
        "3. Coverage & Liability\n"
        "4. Special Notes / Red Flags\n"
        "5. Recommended Next Steps\n\n"
        "Under each heading, write short paragraphs or bullet points."
    ),
    output_key="formatted_summary",
)


critic_agent = LlmAgent(
    name="CriticAgent",
    model=DEFAULT_MODEL,
    description="Reviews the formatted summary for gaps and issues.",
    instruction=(
        "You are a QA reviewer for claim summaries. Review the formatted summary "
        "below for quality and completeness:\n\n"
        "{formatted_summary}\n\n"
        "Identify any missing critical information, ambiguity, unfair language, or "
        "policy/coverage issues that need clarification.\n\n"
        "Return:\n"
        "- A bullet list of issues (if any). If no issues, say 'No material issues found.'\n"
        "- At the end, add a final line in the form:\n"
        "VERDICT: pass\n"
        "or\n"
        "VERDICT: revise"
    ),
    output_key="critique",
)


regeneration_agent = LlmAgent(
    name="RegenerationAgent",
    model=DEFAULT_MODEL,
    description="Produces the final claim summary incorporating critique feedback.",
    instruction=(
        "You are the final writer in the claim summarisation pipeline. You receive:\n"
        "- The formatted summary: {formatted_summary}\n"
        "- The critique: {critique}\n\n"
        "If the critique verdict is 'pass', lightly polish the formatted summary "
        "for clarity and flow.\n"
        "If the verdict is 'revise', rewrite the summary to address the issues, "
        "while preserving factual accuracy.\n\n"
        "Return ONLY the final, ready-to-file claim summary in Markdown."
    ),
    output_key="final_summary",
)


root_agent = SequentialAgent(
    name="ClaimsSequentialPipeline",
    description="Runs the multi-step insurance claim summarisation pipeline.",
    sub_agents=[
        data_ingestion_agent,
        parser_agent,
        summary_agent,
        formatter_agent,
        critic_agent,
        regeneration_agent,
    ],
)


qa_agent = LlmAgent(
    name="ClaimQnAAgent",
    model=DEFAULT_MODEL,
    description="Answers questions about the uploaded claim document.",
    instruction=(
        "You are an insurance claim Q&A assistant.\n\n"
        "You are given the full claim text below:\n\n"
        "{claim_text}\n\n"
        "User question:\n"
        "{user_question}\n\n"
        "Answer the question based ONLY on the claim_text. "
        "If the answer is not clearly stated, say you cannot determine it from the document."
    ),
    output_key="qa_answer",
)
