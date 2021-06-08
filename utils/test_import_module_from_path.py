from utils.import_module_from_path import import_module_from_path
from from_root import from_here


def test_import_module_from_path_with_local_import():
    module = import_module_from_path(
        "test_import_module_from_path_as_local",
        from_here("test_import_module_from_path_dummy.py"),
    )

    assert module.test_string == "this module has been successfully imported"


def test_import_module_from_path_with_global_import():
    module = import_module_from_path(
        "test_import_module_from_path_as_global",
        from_here("test_import_module_from_path_dummy.py"),
    )

    assert module.test_string == "this module has been successfully imported"
