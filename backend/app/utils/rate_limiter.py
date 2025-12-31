"""
Rate Limiting Configuration
Uses slowapi for request rate limiting
"""

from slowapi import Limiter
from slowapi.util import get_remote_address

# Create limiter instance
# Uses IP address as the key for rate limiting
limiter = Limiter(key_func=get_remote_address)

# Rate limit configurations
RATE_LIMITS = {
    "default": "100/minute",       # Default for most endpoints
    "auth": "10/minute",           # Login/register (prevent brute force)
    "training": "20/minute",       # Training operations
    "upload": "10/minute",         # File uploads
    "hf_api": "30/minute",         # HuggingFace API calls
}


def get_rate_limit(category: str = "default") -> str:
    """Get rate limit string for a category"""
    return RATE_LIMITS.get(category, RATE_LIMITS["default"])
