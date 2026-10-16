"""
Core LLM Configuration for DeepTrace

This module provides a centralized way to instantiate the LLM used across all agents.
It uses Google's Gemini models directly via the langchain-google-genai SDK.
"""

import os
from dotenv import load_dotenv
load_dotenv()

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
        stop=stop_after_attempt(2),
        wait=wait_fixed(2),
        retry=retry_if_exception(is_transient_error),
        reraise=True
    )
    def _generate(self, *args, **kwargs) -> ChatResult:
        try:
            return super()._generate(*args, **kwargs)
        except Exception as e:
            if is_transient_error(e):
                print(f"[SmartRetry] Detected transient error {type(e).__name__}. Retrying in 2 seconds...")
            raise e

# ==========================================
# CENTRALIZED LLM CONFIGURATION
# ==========================================
# By keeping the LLM initialization in a single helper function, future agents 
# (Planner, Search, Critic, Writer) will all share the exact same configuration, 
# API keys, and retry logic.

def get_llm(structured_schema=None) -> dict:
    """
    Initializes and returns a configured LangChain Runnable for Google Gemini.
    
    Args:
        structured_schema: Optional Pydantic schema class. If provided, with_structured_output 
                           will be applied to EACH model in the fallback chain to ensure
                           fallbacks work seamlessly even for structured outputs.
                           
    Returns:
        dict: Containing the configured 'llm' and a description of the 'model_used' strategy.
    """
    
    openrouter_key = os.environ.get("OPENROUTER_API_KEY")
    
    api_keys_str = (
        os.environ.get("GOOGLE_API_KEYS")
        or os.environ.get("GOOGLE_API_KEY")
        or os.environ.get("GEMINI_API_KEY")
        or os.environ.get("GEMINI_API_KEYS")
    )
    api_keys = [k.strip() for k in api_keys_str.split(",") if k.strip()] if api_keys_str else []
    
    if not api_keys and not openrouter_key:
        raise ValueError("Neither Google Gemini API key (GOOGLE_API_KEYS / GOOGLE_API_KEY / GEMINI_API_KEY) nor OPENROUTER_API_KEY is set.")
        
    # Instantiate active models for each API key
    models_to_try = [
        "gemini-3.8-flash",
        "gemini-3.5-flash",
        "gemini-flash-latest",
        "gemini-pro-latest"
    ]
    
    llm_instances = []
    for key in api_keys:
        for model_name in models_to_try:
            llm = SmartRetryChatGoogleGenerativeAI(
                model=model_name,
                api_key=key,
                temperature=0.2,
                max_retries=0, # Fail fast on 429 quota exhaustion or 404 to immediately route to the next
                timeout=15.0
            )
            # If structured_schema is provided, wrap the model immediately
            if structured_schema:
                try:
                    llm = llm.with_structured_output(structured_schema)
                except Exception:
                    pass # Some models might not support it, let it fail on invoke
            llm_instances.append(llm)
            
    # Ultimate failover: If Google API is completely down (503ing on all models),
    # seamlessly fall back to OpenRouter (Llama/Mistral) if the key exists.
    if openrouter_key:
        from langchain_openai import ChatOpenAI
        for or_model in ["meta-llama/llama-3.1-8b-instruct", "mistralai/mistral-7b-instruct"]:
            or_llm = ChatOpenAI(
                model=or_model,
                api_key=openrouter_key,
                base_url="https://openrouter.ai/api/v1",
                temperature=0.2,
                max_tokens=4096,
                timeout=30.0,
                max_retries=1
            )
            if structured_schema:
                try:
                    or_llm = or_llm.with_structured_output(structured_schema)
                except Exception:
                    pass
            llm_instances.append(or_llm)
            
    if not llm_instances:
        raise ValueError("No valid LLM instances could be initialized.")
        
    # The first model/key combo is primary; all subsequent are fallbacks
    primary_llm = llm_instances[0]
    fallbacks = llm_instances[1:]
    
    if fallbacks:
        # Chain models together using LangChain's native fallback routing
        robust_llm = primary_llm.with_fallbacks(fallbacks)
        model_used_str = f"Google Gemini ({len(models_to_try)} models) with {len(api_keys)} rotating keys + OpenRouter fallback"
    else:
        robust_llm = primary_llm
        model_used_str = "Single LLM instance"
    
    return {
        "llm": robust_llm,
        "model_used": model_used_str
    }
