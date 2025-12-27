from .openml_adapter import OpenMLAdapter


class Ailerons(OpenMLAdapter):
    def __init__(self):
        super().__init__(data_id=44012, target="goal")