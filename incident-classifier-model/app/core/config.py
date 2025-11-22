"""
Configuration Settings
"""

from pydantic_settings import BaseSettings
from pathlib import Path
from typing import List


class Settings(BaseSettings):
    """Application settings"""
    
    # API Settings
    API_HOST: str = "0.0.0.0"
    API_PORT: int = 8001
    API_RELOAD: bool = True
    
    # Model Settings
    MODEL_PATH: str = "models/incident_classifier.pth"
    CONFIDENCE_THRESHOLD: float = 0.6
    USE_GPU: bool = True
    
    # Image Upload Settings
    MAX_IMAGE_SIZE: int = 10485760  # 10MB
    ALLOWED_EXTENSIONS: str = "jpg,jpeg,png,bmp"
    
    # Logging
    LOG_LEVEL: str = "INFO"
    
    # Project paths
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    
    class Config:
        env_file = ".env"
        case_sensitive = True
    
    @property
    def allowed_extensions_list(self) -> List[str]:
        return [ext.strip() for ext in self.ALLOWED_EXTENSIONS.split(",")]


settings = Settings()
