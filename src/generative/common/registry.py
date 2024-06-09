"""Registry class for registering and initialising classes."""

import enum
from typing import Any

from omegaconf import DictConfig


@enum.unique
class Categories(str, enum.Enum):
    MODELS = "models"


class Registry:
    """Registry class."""

    _registry: dict[tuple[str, str], Any] = {}

    @classmethod
    def get(cls, category: str, key: str) -> Any:
        """Get the registered class.

        Args:
            category: category name
            key: value name

        Returns:
            required class
        """
        if (category, key) not in cls._registry.keys():
            raise ValueError(
                f"Key {key} in category {category} has not been registered.",
            )
        else:
            return cls._registry[(category, key)]

    @classmethod
    def register(cls, category: str, name: str) -> Any:
        """Class method for decorating classes.

        Args:
            category: type of class
            name: specific class
        """

        def inner(new_cls: Any) -> Any:
            cls._registry[(category, name)] = new_cls
            return new_cls

        return inner

    @classmethod
    def build(cls, category: str, name: str, cfg: DictConfig) -> Any:
        """Initialise and return required class.

        Args:
            category: category name
            key: value name
            cfg: config

        Returns:
            initialised class
        """
        return cls.get(category, name)(cfg)
