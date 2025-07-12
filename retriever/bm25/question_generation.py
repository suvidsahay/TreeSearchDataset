import os
from langchain_openai import ChatOpenAI
#from langchain.chat_models.openai import ChatOpenAI
from langchain.schema import HumanMessage

def load_openai_key():
    if not os.getenv("OPENAI_API_KEY"):
        raise Exception("Set your OPENAI_API_KEY!")

def generate_questions(doc1, doc2, num_questions=3):
    prompt = f"""
    Generate {num_questions} multi-hop questions where answers require BOTH documents.

    Document 1: {doc1}

    Document 2: {doc2}

    Only list the questions.
    """
    chat = ChatOpenAI(temperature=0, model_name="gpt-4o")
    response = chat.invoke([HumanMessage(content=prompt)])
    #print(f"\nGenerated Questions for claim: {claim}")
    #print(questions_list)
    print("\nGenerated Questions:")
    print(f"Document 1 snippet: {doc1[:100]}...")
    print(f"Document 2 snippet: {doc2[:100]}...")

    return response.content

