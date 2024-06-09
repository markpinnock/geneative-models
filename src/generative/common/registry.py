import enum
from copy import deepcopy
from typing import Any, Callable, Dict, Optional

from omegaconf import DictConfig


@enum.unique
class Categories(str, enum.Enum):
    BLOCKS = "blocks"


class Registry:
    def __init__(self):
        self._registry = {}

    @classmethod
    def get(cls, category: str, key: str) -> Callable:
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
    def register(cls, category: str, name: str) -> Callable:
        """Class method for decorating classes.

        Args:
            category: type of class
            name: specific class
        """

        def inner(new_cls: Callable) -> Callable:
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
