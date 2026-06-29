from typing import Any


def from_pretrained_prefer_local(loader: Any, model_name: str, cache_dir: str, **kwargs: Any) -> Any:
    try:
        return loader.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            local_files_only=True,
            **kwargs,
        )
    except (OSError, ValueError) as exc:
        print(
            f"Local cache unavailable for {model_name}; falling back to online load. "
            f"Original error: {exc}"
        )
        return loader.from_pretrained(
            model_name,
            cache_dir=cache_dir,
            **kwargs,
        )
