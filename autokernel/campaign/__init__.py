"""Versioned optimization campaigns produced by model runtimes."""

from .runner import (
    build_overnight_prompt,
    parse_agent_command,
    run_campaign,
)
from .types import (
    CAMPAIGN_SCHEMA_VERSION,
    CampaignError,
    CampaignTarget,
    OptimizationCampaign,
    ShapeObservation,
    TensorSignature,
    load_campaign,
    prepare_campaign,
    rank_targets,
    write_optimization_plan,
)

__all__ = [
    "CAMPAIGN_SCHEMA_VERSION",
    "CampaignError",
    "CampaignTarget",
    "OptimizationCampaign",
    "ShapeObservation",
    "TensorSignature",
    "build_overnight_prompt",
    "load_campaign",
    "parse_agent_command",
    "prepare_campaign",
    "rank_targets",
    "run_campaign",
    "write_optimization_plan",
]
