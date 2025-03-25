# -*- coding: utf-8 -*-
import os.path
import yaml


class YAMLConfig:
    def __init__(self, file_path):
        self.file_path = file_path
        self._config = self._load_yaml()

    def _load_yaml(self):
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"YAML file not found：{self.file_path}")
        with open(self.file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def get(self, key, default=None):
        return self._config.get(key, default)

    def update(self, key, value):
        self._config[key] = value
        with open(self.file_path, 'w') as file:
            yaml.dump(self._config, file)

