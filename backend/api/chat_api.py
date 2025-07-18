# backend/api/chat_api.py

from fastapi import APIRouter, Depends, HTTPException, status
from typing import List, Dict, Any, Optional

# Assuming UserProfile and LLMService are available from your backend services
from backend.models.user_models import UserProfile
from backend.middleware.auth_middleware import get_current_user # Assuming this provides UserProfile
from backend.services.llm_service import LLMService, get_llm_service_dependency

router = APIRouter()

@router.post("/completion")
async def chat_completion_endpoint(
    messages: List[Dict[str, str]],
    current_user: UserProfile = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service_dependency),
    temperature: Optional[float] = None,
    llm_provider: Optional[str] = None,
    model_name: Optional[str] = None
):
    """
    Handles basic chat completions without tool usage.
    """
    try:
        response_content = llm_service.chat_completion(
            messages=messages,
            user_profile=current_user,
            temperature=temperature,
            llm_provider=llm_provider,
            model_name=model_name
        )
        return {"response": response_content}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Chat completion failed: {str(e)}")

@router.post("/agent")
async def chat_with_agent_endpoint(
    prompt: str,
    chat_history: List[Dict[str, str]],
    current_user: UserProfile = Depends(get_current_user),
    llm_service: LLMService = Depends(get_llm_service_dependency),
    temperature: Optional[float] = None,
    llm_provider: Optional[str] = None,
    model_name: Optional[str] = None
):
    """
    Handles chat interactions with the LLM agent, enabling tool usage.
    """
    try:
        response_content = await llm_service.chat_with_agent(
            prompt=prompt,
            chat_history=chat_history,
            user_profile=current_user,
            user_provided_temperature=temperature,
            user_provided_llm_provider=llm_provider,
            user_provided_model_name=model_name
        )
        return {"response": response_content}
    except HTTPException as e:
        raise e
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Agent chat failed: {str(e)}")

