import os
from dataclasses import dataclass

@dataclass(frozen=True)
class Settings:
    THRESHOLD: float = float(os.getenv("THRESHOLD", "0.5"))
    ENV: str = os.getenv("ENV", "dev")

settings = Settings()