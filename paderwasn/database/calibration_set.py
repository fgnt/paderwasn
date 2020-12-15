from cached_property import cached_property
from lazy_dataset.database import Database
from paderbox.io import load_json
from paderwasn.paths import calib_set_json_path


class CalibrationDataSet(Database):
    def __init__(self, json_path=calib_set_json_path):
        self._json_path = json_path
        super().__init__()

    @cached_property
    def data(self):
        return load_json(self._json_path)
