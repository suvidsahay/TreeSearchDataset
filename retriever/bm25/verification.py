from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage, SystemMessage
import json
import os
import re

def verify_question(documents, question):
    doc_prompt = f"Passage A:\n{documents[0]}\n\nPassage B:\n{documents[1]}\n"

    # System message for stricter output
    system_msg = SystemMessage(content=(
        "You are an AI verifier. Respond ONLY with valid JSON in the format specified. "
        "No extra text, no explanation, no markdown. Use lowercase booleans."
    ))

    # Short, explicit prompt
    user_prompt = f"""
Given the passages below, answer the following in JSON:

{{
  "question": "{question}",
    - Correct_2_passage: true if BOTH passages are required, false otherwise.
    - Correct_A_passage: true if ONLY passage A is required, false otherwise.
    - Correct_B_passage: true if ONLY passage B is required, false otherwise.
    - Correct_no_passage: true if the question is answerable from common knowledge or NEITHER passage, false otherwise.
  "answer": "<answer using ONLY the passages, or 'Not answerable from passages.'>"
}}

{doc_prompt}
"""

    chat = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o-mini",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    response = chat.invoke([
        system_msg,
        HumanMessage(content=user_prompt)
    ])

    # Robust JSON extraction
    try:
        # Extract the first {...} block
        match = re.search(r'\{.*\}', response.content, re.DOTALL)
        if match:
            json_data = match.group(0)
            result = json.loads(json_data)
        else:
            raise ValueError("No JSON found")
    except Exception:
        result = {
            "question": question,
            "Correct_2_passage": None,
            "Correct_A_passage": None,
            "Correct_B_passage": None,
            "Correct_no_passage": None,
            "answer": "Parsing Error"
        }

    return result

