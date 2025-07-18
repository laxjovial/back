# backend/api/document_api.py

import base64
from fastapi import APIRouter, Depends, HTTPException, UploadFile, File, Form, status
from typing import List, Optional, Any # Ensure Any is imported if used directly
import os
import shutil
import logging

# Assuming UserProfile is available
from backend.models.user_models import UserProfile
from backend.middleware.auth_middleware import get_current_user

# Import the DocumentTools class
from domain_tools.document_tools import DocumentTools

# Import the new dependency functions
from backend.dependencies import (
    get_user_manager,
    get_document_tools_dependency,
)

# Import get_user_tier_capability directly from utils.user_manager for RBAC checks
from utils.user_manager import get_user_tier_capability

router = APIRouter()
logger = logging.getLogger(__name__)


@router.post("/upload", response_model=None) # ADDED response_model=None
async def upload_document(
    file: UploadFile = File(...),
    section: str = Form("general"), # e.g., "general", "finance", "legal", "medical"
    current_user: UserProfile = Depends(get_current_user),
    user_manager: Any = Depends(get_user_manager), # Using Any for user_manager as well for consistency
    document_tools: Any = Depends(get_document_tools_dependency) # Using Any for document_tools
):
    """
    Uploads a document, saves it, and optionally processes it for indexing.
    """
    user_id = current_user.user_id
    logger.info(f"User {user_id} attempting to upload file '{file.filename}' to section '{section}'")

    # RBAC check for document upload capability
    if not get_user_tier_capability(user_id, 'document_upload_enabled', False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Document upload is not enabled for your current tier."
        )

    # Define the upload directory structure: uploads/{user_id}/{section}/
    upload_dir = os.path.join("uploads", user_id, section)
    os.makedirs(upload_dir, exist_ok=True)
    
    file_path = os.path.join(upload_dir, file.filename)

    try:
        # Save the file locally first
        with open(file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)
        
        # Read the file content for base64 encoding
        with open(file_path, "rb") as f:
            file_content_bytes = f.read()
        file_content_base64 = base64.b64encode(file_content_bytes).decode('utf-8')

        # Call the document tool to handle processing/indexing
        process_result_message = await document_tools.document_process_uploaded_document(
            file_name=file.filename,
            file_content_base64=file_content_base64,
            user_token=user_id
        )

        return {
            "message": f"File '{file.filename}' uploaded and processing initiated. Status: {process_result_message}",
            "file_path": file_path
        }
    except Exception as e:
        logger.error(f"Error uploading or processing file for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to upload or process document: {str(e)}")

@router.post("/query", response_model=None) # ADDED response_model=None
async def query_documents(
    query: str = Form(...),
    section: str = Form("general"),
    export: Optional[bool] = Form(False),
    k: Optional[int] = Form(5),
    current_user: UserProfile = Depends(get_current_user),
    user_manager: Any = Depends(get_user_manager), # Using Any for user_manager
    document_tools: Any = Depends(get_document_tools_dependency) # Using Any for document_tools
):
    """
    Queries previously uploaded and indexed documents.
    """
    user_id = current_user.user_id
    logger.info(f"User {user_id} attempting to query documents in section '{section}' with query: '{query}'")

    # RBAC check for document query capability
    if not get_user_tier_capability(user_id, 'document_query_enabled', False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Document querying is not enabled for your current tier."
        )

    try:
        results = await document_tools.document_query_uploaded_docs(
            query=query,
            user_token=user_id,
            section=section,
            export=export,
            k=k
        )
        return {"results": results}
    except Exception as e:
        logger.error(f"Error querying documents for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to query documents: {str(e)}")

@router.post("/summarize_by_path", response_model=None) # ADDED response_model=None
async def summarize_document_by_path_endpoint(
    file_path: str = Form(...),
    current_user: UserProfile = Depends(get_current_user),
    user_manager: Any = Depends(get_user_manager), # Using Any for user_manager
    document_tools: Any = Depends(get_document_tools_dependency) # Using Any for document_tools
):
    """
    Summarizes a document located at a given file path.
    The file path must be accessible by the backend (e.g., in the 'uploads' directory).
    """
    user_id = current_user.user_id
    logger.info(f"User {user_id} attempting to summarize document at path: '{file_path}'")

    # RBAC check for summarization capability
    if not get_user_tier_capability(user_id, 'summarization_enabled', False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Document summarization is not enabled for your current tier."
        )
    
    # Basic path validation to ensure it's within expected upload directories
    expected_prefix = os.path.join("uploads", user_id)
    if not os.path.abspath(file_path).startswith(os.path.abspath(expected_prefix)):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file path. Document must be in your designated upload directory."
        )

    try:
        summary = await document_tools.document_summarize_document_by_path(
            file_path_str=file_path,
            user_token=user_id
        )
        return {"summary": summary}
    except FileNotFoundError:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Document not found at the specified path.")
    except Exception as e:
        logger.error(f"Error summarizing document for user {user_id}: {e}", exc_info=True)
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Failed to summarize document: {str(e)}")

