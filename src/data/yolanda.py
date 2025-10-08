from .openml_adapter import OpenMLAdapter


class Yolanda(OpenMLAdapter):
    def __init__(self, numerical_only: bool):
        super().__init__(data_id=42705, target="101")
        if numerical_only:
            self._remove_categorical()
