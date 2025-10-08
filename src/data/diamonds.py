from .openml_adapter import OpenMLAdapter

class Diamonds(OpenMLAdapter):
    def __init__(self, numeric_only: bool):
        super().__init__(data_id=42225, target="price")
        if numeric_only:
            self._remove_categorical()