"""Cấu hình toàn cục, nạp từ .env (pydantic-settings)."""
from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # --- Detector backend ---
    detector_backend: str = "vllm"          # "vllm" | "ollama"

    # --- vLLM (OpenAI-compatible) ---
    vlm_base_url: str = "http://localhost:8000/v1"
    vlm_model: str = "Qwen/Qwen3-VL-8B-Instruct-AWQ"
    vlm_api_key: str = "EMPTY"

    # --- Ollama ---
    ollama_host: str = "http://localhost:11434"
    ollama_model: str = "qwen3-vl:8b"

    # --- VLM chung ---
    vlm_timeout: int = 600                  # giây/trang (trang dày sinh JSON lâu)
    vlm_max_pixels: int = 2_304_000         # giới hạn pixel ảnh gửi VLM (khớp vLLM max_pixels)
    vlm_temperature: float = 0.0
    vlm_max_output_tokens: int = 6000       # chặn output để 1 trang không sinh vô tận

    # --- Preprocess ---
    render_dpi: int = 300
    do_deskew: bool = True
    deskew_threshold_degrees: float = 0.5

    # --- Detection scope ---
    detect_answers: bool = False        # False = chỉ khoanh vùng câu hỏi trọn vẹn (chưa tách đáp án)

    # --- Box-snap ---
    snap_pad: int = 8
    snap_enabled: bool = True

    # --- Output ---
    output_dir: str = "./output"
    save_vlm_raw: bool = True

    # --- Logging ---
    log_level: str = "INFO"


settings = Settings()
