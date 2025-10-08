from river.datasets import base
from river import stream
from sklearn.datasets import fetch_california_housing

class CalHousing(base.Dataset):
    def __init__(self):
        self.X, self.y = self._fetch()
        super().__init__(
            task=base.REG,
            n_features=len(self.X.columns),
        )

    def _fetch(self):
        return fetch_california_housing(return_X_y=True, as_frame=True)

    def __iter__(self):
        return stream.iter_pandas(
            X=self.X,
            y=self.y,
        )