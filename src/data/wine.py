from .uci_adapter import UCIAdapter
from river import stream

class Wine(UCIAdapter):
    def __init__(self):
        super().__init__(data_id=186)

    def __iter__(self):
        return stream.iter_pandas(
            X=self.X,
            y=self.y["quality"],
        )