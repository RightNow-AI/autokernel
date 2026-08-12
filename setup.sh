#!/bin/bash


# Install uv (if you don't have it)
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and setup
uv sync

# One-time setup: test data + baselines
uv run prepare.py

# Install claude
curl -fsSL https://claude.ai/install.sh | bash
