from typing import List, Optional, Dict, Tuple
from langchain_core.messages import HumanMessage
import os
from prompt_loader import render_prompt

def generate_seed_questions(doc1: str, num_questions: int = 3, chat: Optional[object] = None) -> str:
    prompt = render_prompt(
    "question_generation/generate_seed_questions.j2",
    num_questions=num_questions,
    doc1=doc1,
    )
    if chat is None:
        raise RuntimeError("GENERATION LLM must be injected")

    response = chat.invoke([HumanMessage(content=prompt)])

    # print("\nGenerated Seed Questions, Explanations, and Ground Truth Answers:")
    return response.content

def generate_questions(doc1: str, doc2: str, num_questions: int = 3, chat: Optional[object] = None) -> str:
    # Kept for compatibility, but updated constraints just in case
    prompt = render_prompt(
    "question_generation/generate_questions.j2",
    num_questions=num_questions,
    doc1=doc1,
    doc2=doc2,
    )

    if chat is None:
        raise RuntimeError("GENERATION LLM must be injected")

    response = chat.invoke([HumanMessage(content=prompt)])
    return response.content

def generate_multihop_questions(documents: List[str], num_questions: int = 3, chat: Optional[object] = None) -> str:
    """
    Generates multi-hop questions that require combining information from all provided passages.
    Enforces strict < 5 word answers and reasoning chains.
    """
    num_docs = len(documents)
    if num_docs < 2:
        print("Warning: At least two documents are required to generate multi-hop questions.")
        return ""

    formatted_docs = []
    for i, doc in enumerate(documents):
        formatted_docs.append(f"Document {i + 1}:\n{doc}")
    documents_section = "\n\n---\n\n".join(formatted_docs)

    prompt = render_prompt(
    "question_generation/generate_multihop_questions.j2",
    num_docs=num_docs,
    num_questions=num_questions,
    documents_section=documents_section,
    )

    if chat is None:
        raise RuntimeError("GENERATION LLM must be injected")

    response = chat.invoke([HumanMessage(content=prompt)])
    return response.content

def revise_question(documents: list, failed_question_data: Dict[str, str], minimal_passages_used: List[Tuple[str, str]], naturalness_details: Dict, chat: Optional[object] = None) -> str:
    """
    Revises a question that failed verification (because it didn't use all docs).
    """
    num_docs = len(documents)
    formatted_docs = [f"Document {i + 1}:\n{doc}" for i, doc in enumerate(documents)]
    documents_section = "\n\n---\n\n".join(formatted_docs)

    original_q = failed_question_data.get("question", "N/A")
    used_titles = [title for title, _ in minimal_passages_used]
    used_passages_section = "\n".join([f"- '{title}'" for title in used_titles])
    ld_score = naturalness_details.get("logical_dependency_score", "N/A")
    combines_score = naturalness_details.get("combines_passages_score", "N/A")

    naturalness_feedback = f""" 
    **PREVIOUS ATTEMPT QUALITY SCORES (Out of 5.0):**
    - Logical Dependency Score: {ld_score}
    - Combines Passages Score: {combines_score}
"""

    prompt = render_prompt(
        "question_generation/revise_question.j2",
        num_docs=num_docs,
        naturalness_feedback=naturalness_feedback.strip(),
        original_q=original_q,
        num_required_passages=len(minimal_passages_used),
        used_passages_section=used_passages_section,
        documents_section=documents_section,
    )
    if chat is None:
        raise RuntimeError("GENERATION LLM must be injected")

    response = chat.invoke([HumanMessage(content=prompt)])
    return response.content

def generate_simplified_question(original_question: str, original_answer: str, kept_passages: List[str], num_hops: int, chat: Optional[object] = None) -> str:
    """
    Generates a simpler question using only the kept passages (Top-Down Phase 2).
    Includes strong constraints to prevent 1-hop generation (lazy simplification).
    """
    formatted_docs = []
    for i, doc in enumerate(kept_passages):
        formatted_docs.append(f"Document {i + 1}:\n{doc}")
    documents_section = "\n\n---\n\n".join(formatted_docs)

    prompt = render_prompt(
    "question_generation/generate_simplified_question.j2",
    original_question=original_question,
    original_answer=original_answer,
    num_hops=num_hops,
    documents_section=documents_section
    )
    if chat is None:
        raise RuntimeError("GENERATION LLM must be injected")

    response = chat.invoke([HumanMessage(content=prompt)])
    return response.content