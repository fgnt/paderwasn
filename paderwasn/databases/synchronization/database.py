from lazy_dataset.database import JsonDatabase


class AsyncWASN(JsonDatabase):
    def __init__(self, json_path):
        super().__init__(json_path)

    def get_data_set_scenario_1(self):
        return self.get_dataset(['scenario_1'])

    def get_data_set_scenario_2(self):
        return self.get_dataset(['scenario_2'])

    def get_data_set_scenario_3(self):
        return self.get_dataset(['scenario_3'])

    def get_data_set_scenario_4(self):
        return self.get_dataset(['scenario_4'])
