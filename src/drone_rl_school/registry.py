import importlib

class Registry:
    def __init__(self):
        self._entries = {}

    def register(self, id: str, entry_point: str):
        if id in self._entries:
            raise KeyError(f"{id} already registered")
        self._entries[id] = entry_point

    def make(self, id: str, **kwargs):
        ep = self._entries.get(id)
        if not ep:
            raise KeyError(f"{id} not found")
        module_path, cls_name = ep.split(":")
        module = importlib.import_module(module_path)
        return getattr(module, cls_name)(**kwargs)

    def list(self):
        return list(self._entries)


# Create the two registries we need
env_registry = Registry()
agent_registry = Registry()
