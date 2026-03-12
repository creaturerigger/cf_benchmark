from abc import ABC, abstractmethod


class BaseModel(ABC):
    @abstractmethod
    def forward(self, input):
        pass
