import os

from fastapi import HTTPException, Depends
from fastapi.security import APIKeyHeader

# Get API key from environment variable
API_KEY = os.getenv("API_KEY", "")
# Define API key security scheme for Swagger
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def verify_api_key(api_key: str = Depends(api_key_header)):
    """Verify API key for protected endpoints"""
    if not API_KEY:
        raise HTTPException(status_code=500, detail="API key not configured")
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return True
