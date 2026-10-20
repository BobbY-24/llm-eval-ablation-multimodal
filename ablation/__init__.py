"""Context-ablation runner (A0-A4) for term-meaning prompts over LLM APIs.

Ported from the TermSight design: paired stimuli per item, a strict JSON
response contract with at most one repair, gold labels kept outside the
inference boundary, and paired condition contrasts at scoring time.
"""
