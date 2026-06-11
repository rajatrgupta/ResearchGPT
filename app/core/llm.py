"""
Core LLM Configuration for DeepTrace

This module provides a centralized way to instantiate the LLM used across all agents.
It implements a fault-tolerant fallback strategy using LangChain's built-in 
Runnable fallbacks across multiple free OpenRouter models.
"""

import os
from langchain_openai import ChatOpenAI

# ==========================================
# MODULE CONSTANTS
# ==========================================
# Clean architecture: Extracting the model list to a module-level constant
# makes it easier to manage and update without modifying the core logic inside the function.
FREE_MODELS = [
    "meta-llama/llama-3.1-8b-instruct",
    "mistralai/mistral-7b-instruct",
    "google/gemma-2-9b-it",
    "microsoft/phi-3-mini-128k-instruct"
]

# ==========================================
# CENTRALIZED LLM CONFIGURATION
# ==========================================
# Why centralized LLM configuration is better:
# By keeping the LLM initialization in a single helper function, future agents 
# (Planner, Search, Critic, Writer) will all share the exact same configuration, 
# API keys, and fallback logic. If we need to switch providers, update retry logic,
# or change temperature settings globally, we only change it in one place. 
# This avoids code duplication and ensures consistency across the entire 
# multi-agent system.

# ==========================================
# OPENROUTER FREE MODEL STRATEGY
# ==========================================
# Why model fallback is important:
# We are using free OpenRouter endpoints, which often experience rate limits, 
# sudden timeouts, or API errors due to high demand. Depending on a single model 
# creates a single point of failure. If that model goes down, the entire DeepTrace 
# pipeline stalls and errors out.
#
# What would happen if we depended on only one model:
# If Mistral-7b goes down for 5 minutes, every user query fails immediately. 
# The pipeline crashes mid-flight, potentially stranding research data in 
# the database with a "failed" status.
#
# How production AI systems handle model failures:
# Production systems implement "graceful degradation." They use LLM gateways or 
# internal routers. If the primary, most capable model fails to respond, the 
# system instantly reroutes the payload to a secondary, perhaps slightly smaller, 
# fallback model. This guarantees high availability and ensures the user always 
# receives an answer without manual intervention.

def get_llm() -> dict:
    """
    Initializes and returns a configured LangChain Runnable (ChatOpenAI) with a 
    built-in fallback strategy prioritizing specific free OpenRouter models.
    
    Returns:
        dict: Containing the configured 'llm' and a description of the 'model_used' strategy.
    """
    
    # Fail fast if API key is missing. 
    # Using a dummy key obscures configuration issues and makes debugging harder 
    # when deploying to new environments.
    api_key = os.environ.get("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY environment variable is not set.")
        
    base_url = "https://openrouter.ai/api/v1"
    
    # Recommended OpenRouter headers to improve observability on their dashboard
    # and comply with their usage patterns.
    default_headers = {
        "HTTP-Referer": "https://deeptrace.local", 
        "X-Title": "DeepTrace"
    }
    
    # Initialize all LLMs using modern parameters and explicit constants
    llms = [
        ChatOpenAI(
            model=name,          # Replaced deprecated model_name
            base_url=base_url,   # Replaced deprecated openai_api_base
            api_key=api_key,
            temperature=0.3,
            max_retries=1,       # Fail fast so we can jump to the next fallback
            default_headers=default_headers
        ) for name in FREE_MODELS
    ]
    
    primary_llm = llms[0]
    fallback_llms = llms[1:]
    
    # LangChain's native fallback architecture:
    # If the primary LLM raises an APIError, Timeout, or RateLimit during invoke(),
    # it automatically tries the next LLM in the fallback_llms list.
    robust_llm = primary_llm.with_fallbacks(fallback_llms)
    
    return {
        "llm": robust_llm,
        "model_used": "LangChain Fallback Router (Priority: Llama 3.1 -> Mistral -> Gemma -> Phi)"
    }
