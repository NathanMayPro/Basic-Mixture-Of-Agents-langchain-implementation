# Mixture of Agents (MoA) Implementation

A Python implementation of Mixture of Agents using LangChain, allowing for dynamic interaction between multiple AI models with different roles or strategies.

## Overview

This implementation provides a flexible framework for combining multiple language models in a layered approach, with three main strategies:
- **Role-based**: Models take on specific roles with defined perspectives
- **Temperature-based**: Models use different temperature settings for varied responses
- **Model-based**: Different models are used directly for diverse responses

## Features

- Support for multiple LLM providers (Mistral, OpenAI, Anthropic, Groq)
- Configurable number of layers and models per layer
- Role generation and assignment
- Response aggregation
- Detailed logging and tracing
- Flexible strategy selection

## Installation

```bash
python3 -m venv .venv
.venv/bin/python3 -m pip install -r requirements.txt
```

## Basic usage example:

### Initialize MoA
```python
from moa import MoA
import asyncio
# Initialize MoA
moa = MoA(
    models_layer=["mistral-small", "mistral-medium"], 
    model_roles="mistral-medium",
    aggregator_model="mistral-medium-latest",
    nb_models_by_layer=2,
    layer=2,
    layer_strategy='role'
)
```
### Run with a prompt

```python
result = asyncio.run(moa.run("Your prompt here"))
print(result)
```
### Command Line Usage
```python
python moa.py --user_prompt "Your prompt" \
    --layer 2 \
    --models-layer gpt-4 claude-3-haiku \
    --nb-models-by-layer 2 \
    --model-roles gpt-4 \
    --aggregator-model claude-3-haiku \
    --layer-strategy role
```

## Configuration Options
models_layer: List of models to use in each layer
model_roles: Model used for role generation
aggregator_model: Model used for response aggregation
nb_models_by_layer: Number of models/roles per layer
layer: Number of iteration layers
layer_strategy: Strategy for model interaction ('role', 'temperature', or 'model')
temperature: Temperature setting for model responses
max_tokens: Maximum tokens for model responses
## Logging
The implementation includes comprehensive logging with JSONL format traces stored in ./logs/moa/.
License