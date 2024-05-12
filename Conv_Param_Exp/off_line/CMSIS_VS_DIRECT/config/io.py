import toml
import re


class TOMLConfig:
    def __init__(self, filepath):
        self.filepath = filepath
        self.config = self.load_config()
        self.resolve_references()

    def load_config(self):
        with open(self.filepath, 'r', encoding='utf8') as file:
            return toml.load(file)

    def resolve_references(self):
        # Iterate through all sections and keys, resolving references
        for section in self.config:
            for key in self.config[section]:
                self.config[section][key] = self.resolve_value(self.config[section][key], section)

    def resolve_value(self, value, current_section):
        # Ensure value is a string before attempting to resolve references
        if isinstance(value, str):
            pattern = r"\$\{([^}]+)\}"
            while True:
                match = re.search(pattern, value)
                if not match:
                    break
                ref_path = match.group(1)
                ref_value = self.get_ref_value(ref_path, current_section)
                if isinstance(ref_value, str):
                    value = value.replace(match.group(0), ref_value)
                else:
                    raise ValueError(f"Reference '{ref_path}' cannot be resolved to a string.")
        return value

    def get_ref_value(self, ref_path, current_section):
        parts = ref_path.split('.')
        ref_value = self.config[current_section]
        for part in parts:
            ref_value = ref_value.get(part)
            if ref_value is None:
                # If not found in current section, try to resolve from the whole config
                ref_value = self.config
                for part in parts:
                    ref_value = ref_value.get(part)
                    if ref_value is None:
                        raise ValueError(f"Reference '{ref_path}' not found in configuration.")
                break
        if isinstance(ref_value, dict):
            raise ValueError(f"Reference '{ref_path}' cannot be resolved to a string.")
        return str(ref_value)

    def __getitem__(self, item):
        return self.config[item]

    def get(self, section, key, default=None):
        return self.config.get(section, {}).get(key, default)

    def __repr__(self):
        return '\n'.join([f'[{section}]' + '\n' + '\n'.join([f'{key} = {value}' for key, value in self.config[section].items()]) for section in self.config])