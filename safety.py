from __future__ import annotations
from typing import Dict, Any, List

HIGH_RISK_SIGNS = [
    "insulin overdose",
    "stop taking medication",
    "replace your doctor",
    "cure diabetes",
    "miracle cure",
    "no need dialysis",
]

def basic_safety_scan(text: str) -> Dict[str, Any]:
    """
    Lightweight safety scan for harmful medical directions.
    """
    lower = text.lower()
    hits = [p for p in HIGH_RISK_SIGNS if p in lower]
    return {"ok": len(hits) == 0, "hits": hits}

def medical_disclaimer() -> str:
    return (
        "⚠️ **Medical note:** I can provide nutrition education and meal-planning support, "
        "but this doesn’t replace professional medical advice. If you have urgent symptoms "
        "or complex conditions (pregnancy, severe kidney disease, insulin adjustments, etc.), "
        "please consult a clinician/dietitian."
    )

def enforce_dietary_constraints(plan_text: str, hard_constraints: List[str]) -> Dict[str, Any]:
    """
    Simple constraint check: ensures forbidden items are not suggested.
    hard_constraints examples: ["no grapefruit", "low potassium", "no alcohol"]
    """
    lower = plan_text.lower()
    violated = []
    for c in hard_constraints:
        # naive mapping: treat constraint keyword(s) as must-not-appear
        tokens = [t.strip() for t in c.replace("no ", "").split(",")]
        for tok in tokens:
            if tok and tok in lower and c.lower().startswith("no "):
                violated.append(c)
                break
    return {"ok": len(violated) == 0, "violated": violated}
