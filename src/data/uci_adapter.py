from ucimlrepo import fetch_ucirepo
from river.datasets import base
from river import stream
import abc


class UCIAdapter(base.Dataset):
    def __init__(self, data_id: int = None):
        self.id = data_id
        self.X, self.y = self._fetch()
        super().__init__(
            task=base.REG,
            n_features=len(self.X.columns),
        )

    def _fetch(self):
        # fetch dataset
        abalone = fetch_ucirepo(id=self.id)
        return abalone.data.features, abalone.data.targets

    def _remove_categorical(self):
        self.X = self.X.select_dtypes(include="float64")

    @abc.abstractmethod
    def __iter__(self):
        return stream.iter_pandas(
            X=self.X,
            y=self.y,
        )
