"""
HuggingFace Token Validation
Validates HF tokens against the HuggingFace API
"""

from typing import Optional
import httpx


async def validate_hf_token(token: str) -> dict:
    """
    Validate a HuggingFace token by calling the whoami endpoint

    Args:
        token: HuggingFace API token

    Returns:
        dict with:
            - valid: bool - whether token is valid
            - username: str | None - HF username if valid
            - name: str | None - Display name if valid
            - email: str | None - Email if valid
            - orgs: list[str] - Organizations the user belongs to
            - error: str | None - Error message if invalid
            - can_write: bool - Whether token has write permission
    """
    if not token or not token.strip():
        return {
            "valid": False,
            "error": "Token is empty",
            "username": None,
            "name": None,
            "email": None,
            "orgs": [],
            "can_write": False,
        }

    # Clean the token
    token = token.strip()

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                "https://huggingface.co/api/whoami-v2",
                headers={"Authorization": f"Bearer {token}"},
            )

            if response.status_code == 200:
                data = response.json()

                # Extract organizations
                orgs = []
                if "orgs" in data:
                    orgs = [org.get("name", "") for org in data.get("orgs", [])]

                # Check write permissions
                # Fine-grained tokens have an 'auth' field with permissions
                can_write = True  # Assume write access for now
                if "auth" in data:
                    auth = data["auth"]
                    # Check if it's a read-only token
                    if auth.get("accessToken", {}).get("role") == "read":
                        can_write = False

                return {
                    "valid": True,
                    "username": data.get("name"),
                    "name": data.get("fullname"),
                    "email": data.get("email"),
                    "orgs": orgs,
                    "can_write": can_write,
                    "error": None,
                }
            elif response.status_code == 401:
                return {
                    "valid": False,
                    "error": "Invalid or expired token",
                    "username": None,
                    "name": None,
                    "email": None,
                    "orgs": [],
                    "can_write": False,
                }
            else:
                return {
                    "valid": False,
                    "error": f"HuggingFace API returned status {response.status_code}",
                    "username": None,
                    "name": None,
                    "email": None,
                    "orgs": [],
                    "can_write": False,
                }

    except httpx.TimeoutException:
        return {
            "valid": False,
            "error": "Connection to HuggingFace timed out",
            "username": None,
            "name": None,
            "email": None,
            "orgs": [],
            "can_write": False,
        }
    except Exception as e:
        return {
            "valid": False,
            "error": f"Failed to validate token: {str(e)}",
            "username": None,
            "name": None,
            "email": None,
            "orgs": [],
            "can_write": False,
        }


async def get_token_permissions(token: str) -> dict:
    """
    Get detailed permissions for a HuggingFace token

    Returns:
        dict with permission details
    """
    result = await validate_hf_token(token)

    if not result["valid"]:
        return result

    # Add more detailed permission info
    return {
        **result,
        "permissions": {
            "read_models": True,
            "write_models": result["can_write"],
            "read_datasets": True,
            "write_datasets": result["can_write"],
            "read_spaces": True,
            "write_spaces": result["can_write"],
        },
    }
