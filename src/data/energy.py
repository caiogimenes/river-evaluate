from .uci_adapter import UCIAdapter
from river import stream


class AppliancesEnergy(UCIAdapter):
    def __init__(self, numeric_only: bool):
        super().__init__(data_id=374)
        if numeric_only:
            self._remove_categorical()

    def _remove_categorical(self):
        self.X = self.X.select_dtypes(include=["float64", "int64"])

    def __iter__(self):
        return stream.iter_pandas(
            X=self.X,
            y=self.y["Appliances"],
        )