# rag.py

from __future__ import annotations

from typing import List, Dict, Any, Optional
import json

from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from llm_manager import get_embedding_model


def ehr_to_documents(ehr: Dict[str, Any]) -> List[Document]:
    docs: List[Document] = []

    def add_doc(
        text: str,
        doc_type: str,
        meta: Optional[Dict[str, Any]] = None,
    ) -> None:
        docs.append(
            Document(
                page_content=text,
                metadata={"type": doc_type, **(meta or {})},
            )
        )

    if "diagnoses" in ehr:
        add_doc(
            text="Diagnoses: " + ", ".join(map(str, ehr["diagnoses"])),
            doc_type="diagnoses",
        )

    if "allergies" in ehr:
        add_doc(
            text="Allergies: " + ", ".join(map(str, ehr["allergies"])),
            doc_type="allergies",
        )

    if "medications" in ehr:
        add_doc(
            text="Medications: " + ", ".join(map(str, ehr["medications"])),
            doc_type="medications",
        )

    labs = ehr.get("labs", [])
    if isinstance(labs, list) and labs:
        for lab in labs:
            add_doc(
                text=(
                    f"Lab: {lab.get('name')} = {lab.get('value')} "
                    f"{lab.get('unit','')} on {lab.get('date','unknown')}"
                ),
                doc_type="lab",
                meta={"lab_name": lab.get("name")},
            )

    notes = ehr.get("clinical_notes", [])
    if isinstance(notes, list) and notes:
        for i, note in enumerate(notes, start=1):
            add_doc(
                text=f"Clinical note #{i}: {note}",
                doc_type="note",
            )

    lifestyle = ehr.get("lifestyle", {})
    if isinstance(lifestyle, dict) and lifestyle:
        add_doc(
            text="Lifestyle: " + json.dumps(lifestyle, ensure_ascii=False),
            doc_type="lifestyle",
        )

    if not docs:
        add_doc(
            text="No structured EHR fields provided.",
            doc_type="ehr_empty",
        )

    return docs


def build_ehr_vectorstore(ehr: Dict[str, Any]) -> FAISS:
    emb = get_embedding_model()
    docs = ehr_to_documents(ehr)
    return FAISS.from_documents(docs, emb)


def retrieve_ehr_context(
    vs: FAISS,
    query: str,
    k: int = 6,
) -> List[Document]:
    retriever = vs.as_retriever(search_kwargs={"k": k})

    # ✅ NEW LANGCHAIN WAY:
    # retriever is a Runnable, so call invoke()
    docs = retriever.invoke(query)

    # Some versions return a single Document; normalize to list
    if isinstance(docs, Document):
        return [docs]

    return list(docs)