from .openml_adapter import OpenMLAdapter


class Ailerons(OpenMLAdapter):
    def __init__(self, numeric_only:bool):
        super().__init__(data_id=44012, target="goal")
        if numeric_only:
            self._remove_categorical()