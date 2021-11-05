from abc import ABC, abstractmethod


class AbstractGraphClass(ABC):
    @abstractmethod
    def __init__(self, data, start_node, end_node=None, **kwargs):
        pass

    @abstractmethod
    def get_shortest_path(self):
        pass

    @abstractmethod
    def get_shortest_path_length(self):
        pass
