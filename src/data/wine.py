from .uci_adapter import UCIAdapter
from river import stream

class Wine(UCIAdapter):
    def __init__(self, numerical_only: bool):
        super().__init__(data_id=186)
        if numerical_only:
            self._remove_categorical()

    def __iter__(self):
        return stream.iter_pandas(
            X=self.X,
            y=self.y["quality"],
        )