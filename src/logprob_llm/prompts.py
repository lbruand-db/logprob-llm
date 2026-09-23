"""The single prompt template, used identically at train and inference time.

Keeping train == inference is essential: the answer-token position we read
log-probabilities from must be the same position the model was fine-tuned to
fill (SPECS §5.1). Both the data generator and the serving read import
`build_routing_prompt` so the template can never drift between the two.
"""

from __future__ import annotations

from .labels import TEAMS


def _options_block() -> str:
    return "\n".join(f"{letter}={name}" for letter, name in TEAMS)


def build_routing_prompt(ticket_text: str) -> str:
    """Prompt that asks for a one-letter routing decision.

    Ends with ``"Answer:"`` so the next generated token is the answer letter.
    """
    return (
        "Route the support ticket to exactly one team. "
        "Reply with a single letter and nothing else.\n"
        f"Teams:\n{_options_block()}\n"
        f'Ticket: "{ticket_text.strip()}"\n'
        "Answer:"
    )


def build_generation_prompt(team_letter: str, team_name: str) -> str:
    """Teacher prompt for label-first synthetic data (SPECS §5.1, WI-2).

    We sample the true team first, then ask the teacher to write a realistic
    ticket for it — so the seed label is gold, for free.
    """
    return (
        "You write realistic customer support tickets for a SaaS product.\n"
        f'Write ONE support ticket (2-4 sentences) that clearly belongs to the "{team_name}" team. '
        "Write only the ticket body — no subject line, no team name, no preamble. "
        "Vary tone and specificity; sometimes make it slightly ambiguous but still best handled by "
        f"{team_name}."
    )
