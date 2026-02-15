"""Read Claude Code CLI's OAuth credentials from system storage.

When a user has logged in via `claude` (Claude Code CLI), this module
extracts the stored OAuth access token so nanobot can reuse it as an
Anthropic API key — no separate API key configuration needed.

Supported platforms:
- macOS: reads from Keychain
- Linux: reads from ~/.claude/.credentials.json

Features:
- Automatic token refresh when access token is expired
"""

import json
import platform
import subprocess
import getpass
import time
from pathlib import Path
from loguru import logger

# Claude Code OAuth configuration (from Claude Code CLI source)
TOKEN_URL = "https://platform.claude.com/v1/oauth/token"
CLIENT_ID = "9d1c250a-e61b-44d9-88ed-5944d1962f5e"

# Buffer time before expiration to trigger refresh (5 minutes in ms)
EXPIRY_BUFFER_MS = 300000


def get_claude_code_token() -> str | None:
    """Extract Claude Code CLI's OAuth access token from system storage.

    If the token is expired and a refresh token is available, it will
    attempt to refresh the token automatically.

    Returns the access token string, or None if unavailable.
    """
    system = platform.system()

    if system == "Darwin":
        return _get_token_from_keychain()
    elif system in ("Linux", "Windows"):
        return _get_token_from_credentials_file()
    else:
        logger.debug(f"Claude Code auth extraction not supported on {system}")
        return None


def _get_token_from_keychain() -> str | None:
    """Extract token from macOS Keychain."""
    username = getpass.getuser()

    try:
        result = subprocess.run(
            [
                "security",
                "find-generic-password",
                "-s", "Claude Code-credentials",
                "-a", username,
                "-w",
            ],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            logger.debug("Claude Code credentials not found in keychain")
            return None

        data = json.loads(result.stdout.strip())
        return _extract_token(data, storage_type="keychain")

    except (subprocess.TimeoutExpired, json.JSONDecodeError, Exception) as e:
        logger.debug(f"Failed to read Claude Code credentials from keychain: {e}")
        return None


def _get_token_from_credentials_file() -> str | None:
    """Extract token from ~/.claude/.credentials.json (Linux/Windows)."""
    creds_file = Path.home() / ".claude" / ".credentials.json"

    if not creds_file.exists():
        logger.debug(f"Claude Code credentials file not found: {creds_file}")
        return None

    try:
        data = json.loads(creds_file.read_text())
        return _extract_token(data, storage_type="file")

    except (json.JSONDecodeError, Exception) as e:
        logger.debug(f"Failed to read Claude Code credentials file: {e}")
        return None


def _is_token_expired(expires_at: int | None) -> bool:
    """Check if token is expired or about to expire."""
    if expires_at is None:
        return True
    current_time_ms = int(time.time() * 1000)
    return current_time_ms + EXPIRY_BUFFER_MS >= expires_at


def _refresh_token(refresh_token: str) -> dict | None:
    """Refresh the OAuth token using the refresh token.
    
    Returns the new token data or None if refresh failed.
    """
    try:
        import httpx
    except ImportError:
        logger.debug("httpx not available for token refresh, trying requests")
        try:
            import requests
            return _refresh_token_with_requests(refresh_token)
        except ImportError:
            logger.error("Neither httpx nor requests available for token refresh")
            return None

    return _refresh_token_with_httpx(refresh_token)


def _refresh_token_with_httpx(refresh_token: str) -> dict | None:
    """Refresh token using httpx."""
    import httpx
    
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": CLIENT_ID,
        "scope": "user:inference user:mcp_servers user:profile user:sessions:claude_code",
    }

    try:
        response = httpx.post(
            TOKEN_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30.0,
        )
        
        if response.status_code != 200:
            logger.error(f"Token refresh failed: {response.status_code} - {response.text}")
            return None

        data = response.json()
        logger.info("Successfully refreshed Claude Code OAuth token")
        return data

    except Exception as e:
        logger.error(f"Token refresh request failed: {e}")
        return None


def _refresh_token_with_requests(refresh_token: str) -> dict | None:
    """Refresh token using requests library."""
    import requests
    
    payload = {
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        "client_id": CLIENT_ID,
        "scope": "user:inference user:mcp_servers user:profile user:sessions:claude_code",
    }

    try:
        response = requests.post(
            TOKEN_URL,
            json=payload,
            headers={"Content-Type": "application/json"},
            timeout=30,
        )
        
        if response.status_code != 200:
            logger.error(f"Token refresh failed: {response.status_code} - {response.text}")
            return None

        data = response.json()
        logger.info("Successfully refreshed Claude Code OAuth token")
        return data

    except Exception as e:
        logger.error(f"Token refresh request failed: {e}")
        return None


def _save_credentials_to_file(data: dict) -> bool:
    """Save updated credentials to ~/.claude/.credentials.json."""
    creds_file = Path.home() / ".claude" / ".credentials.json"

    try:
        creds_file.write_text(json.dumps(data))
        logger.debug(f"Saved updated credentials to {creds_file}")
        return True
    except Exception as e:
        logger.error(f"Failed to save credentials: {e}")
        return False


def _save_credentials_to_keychain(data: dict) -> bool:
    """Save updated credentials to macOS Keychain."""
    username = getpass.getuser()
    
    try:
        # Delete existing entry first
        subprocess.run(
            [
                "security",
                "delete-generic-password",
                "-s", "Claude Code-credentials",
                "-a", username,
            ],
            capture_output=True,
            timeout=5,
        )
        
        # Add new entry
        json_data = json.dumps(data)
        result = subprocess.run(
            [
                "security",
                "add-generic-password",
                "-s", "Claude Code-credentials",
                "-a", username,
                "-w", json_data,
            ],
            capture_output=True,
            timeout=5,
        )
        
        if result.returncode != 0:
            logger.error(f"Failed to save to keychain: {result.stderr}")
            return False
            
        logger.debug("Saved updated credentials to keychain")
        return True

    except Exception as e:
        logger.error(f"Failed to save credentials to keychain: {e}")
        return False


def _extract_token(data: dict, storage_type: str = "file") -> str | None:
    """Extract access token from credentials data, refreshing if needed."""
    oauth = data.get("claudeAiOauth", {})
    access_token = oauth.get("accessToken")
    refresh_token = oauth.get("refreshToken")
    expires_at = oauth.get("expiresAt")

    # Check if token is expired
    if _is_token_expired(expires_at):
        logger.info("Claude Code OAuth token is expired, attempting refresh...")
        
        if not refresh_token:
            logger.warning("No refresh token available, cannot refresh")
            return None
        
        # Attempt to refresh
        new_token_data = _refresh_token(refresh_token)
        
        if new_token_data:
            # Update the credentials
            new_access_token = new_token_data.get("access_token")
            new_refresh_token = new_token_data.get("refresh_token", refresh_token)
            expires_in = new_token_data.get("expires_in", 3600)
            new_expires_at = int(time.time() * 1000) + (expires_in * 1000)
            
            # Update oauth data
            oauth["accessToken"] = new_access_token
            oauth["refreshToken"] = new_refresh_token
            oauth["expiresAt"] = new_expires_at
            
            # Parse scopes if provided
            if "scope" in new_token_data:
                oauth["scopes"] = new_token_data["scope"].split()
            
            data["claudeAiOauth"] = oauth
            
            # Save back to storage
            if storage_type == "keychain":
                _save_credentials_to_keychain(data)
            else:
                _save_credentials_to_file(data)
            
            access_token = new_access_token
            logger.info("Successfully refreshed and saved new OAuth token")
        else:
            logger.error("Token refresh failed, returning None")
            return None

    if access_token:
        logger.info("Using Claude Code CLI OAuth token for Anthropic API")
        return access_token

    logger.debug("No accessToken in Claude Code credentials")
    return None
