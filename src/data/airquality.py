from .uci_adapter import UCIAdapter
from river import stream


class AirQuality(UCIAdapter):
    def __init__(self, target: str = "NOx(GT)"):
        super().__init__(data_id=360)
        self.target = target

    def __iter__(self):
        # UCI Air Quality stores the regression target as a column of X
        # (features table), unlike CoverType which reads the label from y.
        # Dropping ``self.target`` from X is required so the label is not also
        # used as a feature.
        return stream.iter_pandas(
            y=self.X[self.target],
            X=self.X.drop(self.target, axis=1),
        )