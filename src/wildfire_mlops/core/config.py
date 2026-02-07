from functools import lru_cache

from pydantic import ConfigDict, Field
from pydantic_settings import BaseSettings

from wildfire_mlops.constants import DEFAULT_CLASS_NAMES, DEFAULT_IMAGE_SIZE


class Settings(BaseSettings):
    model_path: str = Field(default="wildfire_model.pth")
    model_path_custom: str = Field(default="wildfire_model.pth")
    model_path_resnet18: str = Field(default="artifacts/exp3/model_best.pth")
    device: str = Field(default="cpu")
    image_size: int = Field(default=DEFAULT_IMAGE_SIZE)
    class_names: list[str] = Field(default_factory=lambda: DEFAULT_CLASS_NAMES.copy())
    model_arch: str = Field(default="custom_cnn")
    pretrained: bool = Field(default=True)
    reference_stats_path: str = Field(default="artifacts/reference_stats.json")

    def resolve_model_path(self, arch: str | None = None) -> str:
        arch = arch or self.model_arch
        if arch == "resnet18":
            return self.model_path_resnet18
        if arch == "custom_cnn":
            return self.model_path_custom
        return self.model_path

    log_level: str = Field(default="INFO")

    api_host: str = Field(default="0.0.0.0")
    api_port: int = Field(default=8000)

    streamlit_host: str = Field(default="0.0.0.0")
    streamlit_port: int = Field(default=8501)

    model_arch: str = Field(default="custom_cnn")

    model_config = ConfigDict(env_prefix="WILDFIRE_")


@lru_cache
def get_settings() -> Settings:
    return Settings()
