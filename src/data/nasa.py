from .openml_adapter import OpenMLAdapter

class NASA(OpenMLAdapter):
    def __init__(self, numerical_only: bool):
        super().__init__(data_id=42821, target="class")
        if numerical_only:
            self._remove_categorical()