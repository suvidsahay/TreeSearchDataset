from langchain_openai import ChatOpenAI
from langchain.schema import HumanMessage
import json
import os

def verify_question(documents, question):
    doc_prompt = f"""
Passage A:
{documents[0]}

Passage B:
{documents[1]}
"""

    VERIFICATION_PROMPT = f"""
You are an AI verifier. Follow these instructions exactly:

1. Determine:
   - Correct_2_passage: true/false
   - Correct_A_passage: true/false
   - Correct_B_passage: true/false
   - Correct_no_passage: true/false

2. Provide the ground truth answer using ONLY the passages.
   - If not answerable, respond: "Not answerable from passages."

IMPORTANT:
- Respond ONLY in valid JSON.
- No extra text.
- Use lowercase booleans.
- Format:

{{
  "question": "{question}",
  "Correct_2_passage": true/false,
  "Correct_A_passage": true/false,
  "Correct_B_passage": true/false,
  "Correct_no_passage": true/false,
  "answer": "<your answer>"
}}

Process this:

{doc_prompt}
"""

    chat = ChatOpenAI(
        temperature=0,
        model_name="gpt-4o",
        openai_api_key=os.getenv("OPENAI_API_KEY")
    )

    response = chat.invoke([HumanMessage(content=VERIFICATION_PROMPT)])

    # Smart JSON extraction
    try:
        json_start = response.content.find('{')
        json_data = response.content[json_start:]
        result = json.loads(json_data)
    except json.JSONDecodeError:
        result = {
            "question": question,
            "Correct_2_passage": None,
            "Correct_A_passage": None,
            "Correct_B_passage": None,
            "Correct_no_passage": None,
            "answer": "Parsing Error"
        }

    return result

