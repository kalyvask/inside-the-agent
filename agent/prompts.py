"""
System prompt + structured action format for the browser agent.

We use a tight JSON action schema so the parser is deterministic. Llama 3.1-8B
is good enough to follow it consistently when prompted clearly.
"""

SYSTEM_PROMPT = """You are a browser agent. You complete shopping and form tasks on web storefronts.

You will be given:
  - A goal (one sentence)
  - The current page's relevant elements (buttons, links, inputs, text)

You must respond with EXACTLY ONE JSON object on a single line, no extra text:

  {"action": "click",    "target": "<element_id or visible label>"}
  {"action": "type",     "target": "<input_id>",     "text": "<value>"}
  {"action": "scroll",   "direction": "down|up"}
  {"action": "navigate", "url": "<absolute_url>"}
  {"action": "submit",   "target": "<form_id>"}
  {"action": "done",     "reason": "<short reason>"}

Rules:
  - Output exactly one JSON object. No prose. No code fence.
  - Always pick the action that most directly advances the goal.
  - Use "done" when you believe the goal is complete.
  - Do not invent element IDs not present on the page."""


def build_user_prompt(goal: str, page_summary: str, history: list[dict] | None = None) -> str:
    """Compose the per-step user message."""
    lines = [f"GOAL: {goal}", "", "PAGE:", page_summary]
    if history:
        lines.append("")
        lines.append("PREVIOUS ACTIONS:")
        for i, h in enumerate(history[-5:], 1):  # last 5 only
            lines.append(f"  {i}. {h}")
    return "\n".join(lines)


def build_chat_prompt(goal: str, page_summary: str, history: list[dict] | None = None) -> str:
    """Format as a Llama 3.1 instruction prompt."""
    user = build_user_prompt(goal, page_summary, history)
    # Llama 3.1 chat template
    return (
        f"<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n"
        f"{SYSTEM_PROMPT}<|eot_id|>"
        f"<|start_header_id|>user<|end_header_id|>\n\n"
        f"{user}<|eot_id|>"
        f"<|start_header_id|>assistant<|end_header_id|>\n\n"
    )
