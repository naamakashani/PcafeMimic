# -*- coding: utf-8 -*-
"""
Created on Tue Dec  3 10:38:42 2024

@author: mirac
"""

import json
import os

def load_config(username):
    with open("base_config.json", "r", encoding="utf-8") as f:
        base_config = json.load(f)

    user_config_file = f"user_config_{username}.json"
    if os.path.exists(user_config_file):
        with open(user_config_file, "r", encoding="utf-8") as f:
            user_config = json.load(f)
        base_config.update(user_config)
    return base_config

# Example usage:
config = load_config("naama")
print(config["user_specific_setting"])  # Output: alice_value