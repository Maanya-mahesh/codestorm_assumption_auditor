CONCLUSION_EXTRACTION_PROMPT = """
Read this research paper text and identify the 1-2 main conclusions
the authors claim their work proves or demonstrates.

Return ONLY a JSON object:
{{
  "main_conclusion": "one sentence summarizing the paper's core claim",
  "secondary_conclusion": "one sentence for second claim, or empty string"
}}

Paper text:
{text}
"""

EXPLICIT_ASSUMPTION_PROMPT = """
Read this research paper chunk carefully.

List every assumption that is EXPLICITLY stated — things the authors
directly acknowledge they assume, often signaled by words like:
"we assume", "assuming", "we consider", "under the assumption",
"we simplify by", "for simplicity", "we ignore".

For each explicit assumption, return a JSON object:
{{
  "assumption": "one sentence describing what is assumed",
  "category": "data",
  "quote": "verbatim short phrase from the chunk where this is stated",
  "explicit": true
}}

Category must be one of: data, mathematical, scope, computational, worldview, experimental

Return a JSON array. If no explicit assumptions found, return [].

Chunk:
{text}
"""

IMPLICIT_ASSUMPTION_PROMPT = """
You are an adversarial peer reviewer. Read this research paper chunk.

Find assumptions the authors NEVER STATE but silently rely on.
Look for:
- Methods that only work under unstated data conditions
- Mathematical steps that require unstated preconditions
- Claims generalized beyond what experiments show
- Computational steps assuming ideal resources
- Worldview priors never questioned

For each implicit assumption return:
{{
  "assumption": "one sentence — what they silently assume",
  "category": "data",
  "evidence": "which part of the chunk relies on this assumption",
  "detection_reasoning": "brief explanation of why this assumption is hiding here",
  "explicit": false
}}

Category must be one of: data, mathematical, scope, computational, worldview, experimental

Return a JSON array. If none found, return [].

Chunk:
{text}
"""

CRITICALITY_PROMPT = """
Given this assumption from a research paper:
Assumption: "{assumption}"
Category: "{category}"
Main conclusion of the paper: "{conclusion}"

Answer these two things:

1. CRITICALITY — If this assumption is violated in the real world,
   the paper's conclusion would:
   - "collapse" → conclusion is completely invalid (score: 3)
   - "weaken"   → conclusion holds partially but loses strength (score: 2)
   - "survive"  → conclusion still valid with minor degradation (score: 1)

2. REAL_WORLD_BRIDGE — Give ONE concrete real-world scenario where
   this assumption is violated.

Return ONLY this JSON:
{{
  "criticality": "collapse",
  "criticality_score": 3,
  "criticality_reasoning": "one sentence explaining why",
  "real_world_bridge": "one sentence concrete scenario where assumption fails"
}}
"""

LAYMAN_PROMPT = """
Explain this research paper assumption in plain English for someone
with no technical background. Use a simple everyday analogy if possible.
Keep it to 2 sentences maximum.

Assumption: "{assumption}"
Category: "{category}"

Plain English explanation:
"""