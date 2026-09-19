"""
Core LLM Configuration for DeepTrace

This module provides a centralized way to instantiate the LLM used across all agents.
It uses Google's Gemini models directly via the langchain-google-genai SDK.
"""

import os
from langchain_google_genai import ChatGoogleGenerativeAI
from tenacity import retry, stop_after_attempt, wait_fixed, retry_if_exception
from langchain_core.outputs import ChatResult

def is_transient_error(exception: BaseException) -> bool:
    """Check if the error is a temporary 503/504 server spike rather than a 429 quota."""
    error_str = str(exception).lower()
    return "503" in error_str or "504" in error_str or "unavailable" in error_str or "timeout" in error_str

class SmartRetryChatGoogleGenerativeAI(ChatGoogleGenerativeAI):
    """
    Subclass that applies an explicit tenacity retry only for transient 503/504 errors.
    This allows max_retries=0 to quickly failover on 429 quota errors, while still
    protecting the pipeline from random temporary connection spikes.
    """
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_fixed(3),
        retry=retry_if_exception(is_transient_error),
        reraise=True
    )
    def _generate(self, *args, **kwargs) -> ChatResult:
        try:
            return super()._generate(*args, **kwargs)
        except Exception as e:
            if is_transient_error(e):
                print(f"[SmartRetry] Detected transient error {type(e).__name__}. Retrying in 3 seconds...")
            raise e

# ==========================================
# CENTRALIZED LLM CONFIGURATION
# ==========================================
# By keeping the LLM initialization in a single helper function, future agents 
# (Planner, Search, Critic, Writer) will all share the exact same configuration, 
# API keys, and retry logic.

def get_llm() -> dict:
    """
    Initializes and returns a configured LangChain Runnable for Google Gemini.
    
    Returns:
        dict: Containing the configured 'llm' and a description of the 'model_used' strategy.
    """
    
    # Fetch the comma-separated list of keys, falling back to singular for backward compatibility
    api_keys_str = os.environ.get("GOOGLE_API_KEYS") or os.environ.get("GOOGLE_API_KEY")
    if not api_keys_str:
        raise ValueError("GOOGLE_API_KEYS or GOOGLE_API_KEY environment variable is not set.")
        
    api_keys = [k.strip() for k in api_keys_str.split(",") if k.strip()]
    
    if not api_keys:
        raise ValueError("No valid API keys found in the environment variables.")
        
    # Instantiate a model for each API key using our custom retry subclass
    llm_instances = []
    for key in api_keys:
        llm = SmartRetryChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            api_key=key,
            temperature=0.2,
            max_retries=0, # Fail fast on 429 quota exhaustion to immediately route to the next key
            timeout=15.0
        )
        llm_instances.append(llm)
        
    # The first key is the primary LLM; all subsequent keys are fallbacks
    primary_llm = llm_instances[0]
    fallbacks = llm_instances[1:]
    
    if fallbacks:
        # Chain models together using LangChain's native fallback routing
        robust_llm = primary_llm.with_fallbacks(fallbacks)
        model_used_str = f"Google Gemini (1.5-flash) with {len(api_keys)} rotating keys (SmartRetry enabled)"
    else:
        robust_llm = primary_llm
        model_used_str = "Google Gemini (1.5-flash) single key (SmartRetry enabled)"
    
    return {
        "llm": robust_llm,
        "model_used": model_used_str
    }
