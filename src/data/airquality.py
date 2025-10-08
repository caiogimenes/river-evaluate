from .uci_adapter import UCIAdapter
from river import stream

class AirQuality(UCIAdapter):
    def __init__(self, numeric_only: bool, target:str = "NOx(GT)"):
        super().__init__(data_id=360)
        self.target = target
        if numeric_only:
            self._remove_categorical()

    def _remove_categorical(self):
        self.X = self.X.select_dtypes(include=["float64", "int64"])

    def __iter__(self):
        return stream.iter_pandas(
            y=self.X[self.target],
            X=self.X.drop(self.target, axis=1),
        )