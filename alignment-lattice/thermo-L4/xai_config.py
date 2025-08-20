# xAI API Configuration
# This file contains the API key for xAI integration
# Load from environment variables for security

import os
from pathlib import Path

# Try to load from .env file if it exists
env_path = Path(__file__).parent.parent.parent / '.env'
if env_path.exists():
    with open(env_path, 'r') as f:
        for line in f:
            if line.strip() and not line.startswith('#'):
                key, value = line.strip().split('=', 1)
                os.environ[key] = value

# Load configuration from environment variables
XAI_API_KEY = os.getenv('XAI_API_KEY')
XAI_API_URL = os.getenv('XAI_API_URL', 'https://api.x.ai/v1/chat/completions')
XAI_MODEL = os.getenv('XAI_MODEL', 'grok-4-latest')

# Grok-4 specific configuration
XAI_MAX_TOKENS = int(os.getenv('XAI_MAX_TOKENS', '2000'))  # Higher default for Grok-4 reasoning
XAI_TEMPERATURE = float(os.getenv('XAI_TEMPERATURE', '0.7'))
XAI_TOP_P = float(os.getenv('XAI_TOP_P', '0.9'))

# L4 testing specific parameters
XAI_L4_SYSTEM_PROMPT = os.getenv('XAI_L4_SYSTEM_PROMPT', 
    "You are an AI assistant that can perform mathematical simulations and report results.")

# Validate required configuration
if not XAI_API_KEY:
    raise ValueError("XAI_API_KEY not found in environment variables or .env file")

print(f"✅ xAI Configuration loaded successfully")
print(f"   Model: {XAI_MODEL}")
print(f"   Max Tokens: {XAI_MAX_TOKENS}")
print(f"   Temperature: {XAI_TEMPERATURE}") 