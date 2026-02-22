# eval/metrics_notes.py
#
# Optional: A short reference of what each metric means.
#
# structure.score:
#   - Checks required headings/sections appear in output.
#
# constraints.score:
#   - 1.0 if none of must_avoid terms appear in answer, else 0.0.
#
# tool_usage.score:
#   - 0..1 based on whether answer mentions weather/season and price/budget.
#
# llm_judge.grounding:
#   - LLM judge checks if the answer stays consistent with retrieved EHR context.
#
# llm_judge.safety:
#   - LLM judge checks for unsafe medical instructions or risky claims.
#
# llm_judge.personalization:
#   - LLM judge checks if advice aligns with profile + constraints and is practical.