from .uci_adapter import UCIAdapter
from river import stream

class AirQuality(UCIAdapter):
    def __init__(self, target:str = "NOx(GT)"):
        super().__init__(data_id=360)
        self.target = target

    def __iter__(self):
        return stream.iter_pandas(
            y=self.X[self.target],
            X=self.X.drop(self.target, axis=1),
        )