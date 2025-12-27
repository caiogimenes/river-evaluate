from .uci_adapter import UCIAdapter
from river import stream

class CoverType(UCIAdapter):
    def __init__(self, target:str = "Cover_Type"):
        super().__init__(data_id=31)
        self.target = target

    def __iter__(self):
        return stream.iter_pandas(
            y=self.y[self.target],
            X=self.X,
        )