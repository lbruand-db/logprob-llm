"""OSS log-prob LLM: calibrated, typed outputs over a closed answer space.

See SPEC/SPECS.md for the full design. Milestone 1 implements the `Choice`
primitive for support-ticket routing.
"""

__all__ = ["labels", "prompts", "calibrate", "serving_read"]
