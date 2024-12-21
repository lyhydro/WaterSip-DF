# -*- coding: utf-8 -*-
import os.path
import yaml


class YAMLConfig:
    def __init__(self, file_path):
        self.file_path = file_path
        self._config = self._load_yaml()

    def _load_yaml(self):
        """加载YAML文件内容"""
        if not os.path.exists(self.file_path):
            raise FileNotFoundError(f"YAML文件未找到：{self.file_path}")
        with open(self.file_path, 'r', encoding='utf-8') as file:
            return yaml.safe_load(file)

    def get(self, key, default=None):
        """获取配置项的值"""
        return self._config.get(key, default)

    def update(self, key, value):
        """更新配置项的值并保存"""
        self._config[key] = value
        with open(self.file_path, 'w') as file:
            yaml.dump(self._config, file)

