# api/auth.py

import os
import secrets
from fastapi import Header, HTTPException, status
from dotenv import load_dotenv

load_dotenv()

# ----------------------------
# CONFIG
# ----------------------------

raw_keys = os.getenv("API_KEYS")

if not raw_keys:
    raise RuntimeError("API_KEYS is missing in environment variables")

API_KEYS = {key.strip() for key in raw_keys.split(",") if key.strip()}

if len(API_KEYS) == 0:
    raise RuntimeError("API_KEYS is empty after parsing")


# ----------------------------
# AUTH VALIDATION
# ----------------------------

def verify_api_key(x_api_key: str = Header(default=None)):

    if x_api_key is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Missing API key"
        )

    for key in API_KEYS:
        if secrets.compare_digest(x_api_key, key):
            return True

    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Invalid API key"
    )