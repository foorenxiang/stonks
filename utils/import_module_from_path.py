import logging
import importlib.util

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def import_module_from_path(module_name, module_path, is_import_as_local=False):
    """
    module_name: name that module should be imported as
    module_path: path to module
    """
    if module_name and module_path:
        try:
            spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            if is_import_as_local:
                logger.info(
                    f"Imported {module_name} from {module_path} as local variable"
                )
                return module
            globals()[module_name] = module
            logger.info(f"Imported {module_name} from {module_path} into globals")
        except Exception:
            logger.error(f"Failed to import {module_name} from {module_path}")
            raise


if __name__ == "__main__":
    from from_root import from_root

    module = import_module_from_path(
        "test_import_module_from_path_as_global",
        from_root("utils/test_import_module_from_path_dummy.py"),
    )

    assert module.test_string == "this module has been successfully imported"
