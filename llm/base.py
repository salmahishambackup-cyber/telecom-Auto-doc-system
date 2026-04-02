from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    @abstractmethod
    def generate(self, prompt: str, **kwargs) -> str: ...

    @abstractmethod
    def generate_structured(self, prompt: str, schema: type, **kwargs) -> dict: ...

    @abstractmethod
    def generate_with_confidence(self, prompt: str, **kwargs) -> tuple[str, float]: ...
