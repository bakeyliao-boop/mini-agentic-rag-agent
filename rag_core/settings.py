"""项目环境配置的读取与基础校验。"""

from collections.abc import Mapping
from pathlib import Path

from dotenv import dotenv_values


def load_settings_from_env(project_root: Path) -> dict[str, str]:
    """从项目根目录的 .env 读取非空字符串配置。"""

    env_path = project_root / ".env"
    if not env_path.is_file():
        raise FileNotFoundError(f"environment file does not exist: {env_path}")

    loaded_values = dotenv_values(env_path)
    return {
        name: value
        for name, value in loaded_values.items()
        if isinstance(value, str)
    }


def _required_setting(
    settings: Mapping[str, str],
    name: str,
) -> str:
    """读取必需配置，并拒绝缺失或空字符串。"""

    value = settings.get(name)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"missing required setting: {name}")
    return value.strip()
