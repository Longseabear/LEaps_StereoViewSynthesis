from importlib import import_module

def get_instance_from_name(module_path, class_name, config):
    m = import_module(module_path, class_name)
    return m, getattr(m, class_name)(config)

