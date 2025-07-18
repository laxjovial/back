# backend/dependencies.py

from fastapi import Depends
from typing import Any # Ensure Any is imported

# Import the actual classes
from utils.user_manager import UserManager
from database.firestore_manager import FirestoreManager
from shared_tools.cloud_storage_utils import CloudStorageUtilsWrapper
from shared_tools.vector_utils import VectorUtilsWrapper
from backend.services.api_usage_service import ApiUsageService

# Import domain_tools.document_tools.DocumentTools
from domain_tools.document_tools import DocumentTools

# Import singletons/global functions
from config.config_manager import config_manager
from utils.analytics_tracker import log_event # Assuming this is the log_event function

# --- Core Service Dependency Functions ---

def get_user_manager() -> Any:
    """Provides a UserManager instance."""
    return UserManager()

def get_firestore_manager() -> Any:
    """Provides a FirestoreManager instance."""
    # FirestoreManager might need firebase_admin.firestore.client() or similar
    # For now, assuming it can be instantiated directly.
    # If it needs a Firebase app, that should be initialized in main.py and passed here.
    return FirestoreManager()

def get_cloud_storage_utils() -> Any:
    """Provides a CloudStorageUtilsWrapper instance."""
    bucket_name = config_manager.get_secret("gcs_bucket_name")
    if not bucket_name:
        raise ValueError("Google Cloud Storage bucket name (gcs_bucket_name) not found in secrets.")
    return CloudStorageUtilsWrapper(bucket_name=bucket_name)

def get_vector_utils_wrapper(
    # These parameters are now explicitly typed as Any to prevent Pydantic validation
    firestore_manager: Any = Depends(get_firestore_manager),
    cloud_storage_utils: Any = Depends(get_cloud_storage_utils)
) -> Any:
    """Provides a VectorUtilsWrapper instance with its dependencies."""
    return VectorUtilsWrapper(
        firestore_manager=firestore_manager,
        cloud_storage_utils=cloud_storage_utils,
        config_manager=config_manager,
        log_event_func=log_event
    )

def get_api_usage_service(
    # ApiUsageService might also need dependencies.
    # Assuming it can be instantiated with basic dependencies or none for now.
    firestore_manager: Any = Depends(get_firestore_manager),
    user_manager: Any = Depends(get_user_manager)
) -> Any:
    """Provides an ApiUsageService instance."""
    return ApiUsageService(
        firestore_manager=firestore_manager,
        user_manager=user_manager,
        config_manager=config_manager # config_manager is a singleton
    )

# --- Domain-Specific Tool Dependency Functions ---

def get_document_tools_dependency(
    # Parameters for DocumentTools are also typed as Any
    vector_utils_wrapper: Any = Depends(get_vector_utils_wrapper),
    firestore_manager: Any = Depends(get_firestore_manager),
    cloud_storage_utils: Any = Depends(get_cloud_storage_utils)
) -> Any: # Return type for DocumentTools instance is also Any
    """
    FastAPI dependency that provides a DocumentTools instance with its required dependencies.
    """
    return DocumentTools(
        vector_utils_wrapper=vector_utils_wrapper,
        config_manager=config_manager,
        firestore_manager=firestore_manager,
        cloud_storage_utils=cloud_storage_utils,
        log_event_func=log_event
    )

