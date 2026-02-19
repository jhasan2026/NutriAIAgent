from __future__ import annotations

from typing import List, Dict, Any, Optional
import json
import uuid

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from llm_manager import get_embedding_model

def ehr_to_documents(ehr: Dict[str, Any]) -> List[Document]:
    """
    Convert structured EHR JSON into chunkable Documents.
    You can extend the schema as needed.
    """
    docs: List[Document] = []

    def add_doc(text: str, doc_type: str, meta: Optional[Dict[str, Any]] = None):
        docs.append(Document(page_content=text, metadata={"type": doc_type, **(meta or {})}))

    # Common fields
    if "diagnoses" in ehr:
        add_doc("Diagnoses: " + ", ".join(map(str, ehr["diagnoses"])), "diagnoses")
    if "allergies" in ehr:
        add_doc("Allergies: " + ", ".join(map(str, ehr["allergies"])), "allergies")
    if "medications" in ehr:
        add_doc("Medications: " + ", ".join(map(str, ehr["medications"])), "medications")

    # Labs
    labs = ehr.get("labs", [])
    if isinstance(labs, list) and labs:
        for lab in labs:
            # expected lab: {"name":"HbA1c","value":7.2,"unit":"%","date":"2025-01-10"}
            add_doc(
                f"Lab: {lab.get('name')} = {lab.get('value')} {lab.get('unit','')} on {lab.get('date','unknown')}",
                "lab",
                {"lab_name": lab.get("name")}
            )

    # Notes
    notes = ehr.get("clinical_notes", [])
    if isinstance(notes, list) and notes:
        for i, note in enumerate(notes, start=1):
            add_doc(f"Clinical note #{i}: {note}", "note")

    # Lifestyle
    lifestyle = ehr.get("lifestyle", {})
    if isinstance(lifestyle, dict) and lifestyle:
        add_doc("Lifestyle: " + json.dumps(lifestyle, ensure_ascii=False), "lifestyle")

    # If empty, still provide something
    if not docs:
        add_doc("No structured EHR fields provided.", "ehr_empty")

    return docs


def build_ehr_vectorstore(ehr: Dict[str, Any]) -> FAISS:
    emb = get_embedding_model()
    docs = ehr_to_documents(ehr)
    return FAISS.from_documents(docs, emb)


def retrieve_ehr_context(vs: FAISS, query: str, k: int = 6) -> List[Document]:
    retriever = vs.as_retriever(search_kwargs={"k": k})
    return retriever.get_relevant_documents(query)
