from pydantic_settings import BaseSettings
from enum import Enum
from typing import Optional

class Environment(str, Enum):
    LOCAL = "LOCAL"
    DEV = "DEV"
    PROD = "PROD"

class Settings(BaseSettings):
    ENV: Environment = Environment.LOCAL
    HF_SPACE: str = "your-user/your-space"
    HF_TOKEN: Optional[str] = None
    BACKEND_URL: Optional[str] = None
    
    class Config:
        env_file = ".env"
        extra = "ignore"

settings = Settings()
