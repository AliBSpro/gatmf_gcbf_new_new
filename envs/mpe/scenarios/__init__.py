import importlib.util
import os.path as osp

def load(name):
    pathname = osp.join(osp.dirname(__file__), name)
    spec = importlib.util.spec_from_file_location("", pathname)
    if spec is None:
        raise ImportError(f"Could not load spec from {pathname}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module



