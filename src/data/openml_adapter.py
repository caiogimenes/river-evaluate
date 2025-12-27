import openml
from river.datasets import base
from river import stream


class OpenMLAdapter(base.Dataset):
    def __init__(self, target: str, data_id: int = None):
        self.id = data_id
        self.target = target
        self.X, self.y = self._fetch()
        super().__init__(
            task=base.REG,
            n_features=len(self.X.columns),
        )

    def _fetch(self):
        # fetch dataset
        dataset = openml.datasets.get_dataset(self.id)
        X, y, _, _ = dataset.get_data(self.target)
        return X, y

    def __iter__(self):
        return stream.iter_pandas(
            X=self.X,
            y=self.y,
        )
