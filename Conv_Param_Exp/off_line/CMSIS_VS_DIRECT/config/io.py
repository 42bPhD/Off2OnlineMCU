import toml
import re
class TOMLConfig:
    def __init__(self, filepath):
        self.filepath = filepath
        self.config = self.load_config()
        self.resolve_references()

    def load_config(self):
        with open(self.filepath, 'r') as file:
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
                value = value.replace(match.group(0), ref_value)
        return value

    def get_ref_value(self, ref_path, current_section):
        parts = ref_path.split('.')
        try:
            # Attempt to resolve reference from the current section or globally
            ref_value = self.config[current_section]
            for part in parts:
                if part in ref_value:
                    ref_value = ref_value[part]
                else:
                    # If not found in current section, try to resolve from the whole config
                    ref_value = self.config
                    for part in parts:
                        ref_value = ref_value[part]
            if isinstance(ref_value, dict):
                raise ValueError(f"Reference '{ref_path}' cannot be resolved to a string.")
            return str(ref_value)
        except KeyError:
            raise ValueError(f"Reference '{ref_path}' not found in configuration.")

    def __getitem__(self, item):
        return self.config[item]

    def get(self, section, key, default=None):
        return self.config.get(section, {}).get(key, default)