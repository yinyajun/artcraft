import importlib
import importlib.util
import pkgutil


def load_processors(module_name: str):
    tasks = importlib.util.find_spec(module_name)
    processors = {}
    for m in pkgutil.iter_modules(tasks.submodule_search_locations):
        sub_module = importlib.import_module(f"{module_name}.{m.name}")
        process_fn = getattr(sub_module, "process", None)
        ui_fn = getattr(sub_module, "ui", None)
        if process_fn is None or ui_fn is None:
            continue
        processors[m.name] = {"process": process_fn, "ui": ui_fn}
    return processors


