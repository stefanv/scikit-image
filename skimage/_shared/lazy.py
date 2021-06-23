import importlib
import importlib.util
import os
import sys


def package_structure(package_name, subpackages=None, subpkg_attrs=None):
    """Lazily load subpackages, functions and attributes to this package.

    Typically, packages import modules, subpackages and attributes as follows::

      import first_module
      import another_module

      from .foo import someattr

    The idea of the lazy package is to replace the `__init__.py` module's
    `__getattr__`, `__dir__`, and `__all__` attributes such that all
    imports work exactly the way they normally would, except that the
    actual import is delayed until the resulting module object is first used.

    The typical way to call this function, replacing the above imports, is::

      __getattr__, __lazy_dir__, __all__ = lazy.package_structure(
        __name__,
        ['first_module', 'another_module'],
        {'foo': 'someattr'}
      )

    This functionality requires Python 3.7 or higher.

    Parameters
    ----------
    package_name : str
        Typically use __name__. The name of the package we are creating.
    subpackages : set
        List of subpackages and/or modules to attach.
    subpkg_attrs : dict
        Dictionary keyed by subpackage name to a list of attributes.
        These attributes are created the first time the subpackage is used.
        Note that just like `foo` does not appear in the package in the
        example above (only someattr), the subpackage name does not appear
        in the package structured by this function.

    Returns
    -------
    __getattr__, __dir__, __all__

    """
    if subpkg_attrs is None:
        subpkg_attrs = {}

    if subpackages is None:
        subpackages = set()
    else:
        subpackages = set(subpackages)

    attr_to_packages = {
        attr: pkg for pkg, attrs in subpkg_attrs.items() for attr in attrs
    }

    __all__ = list(subpackages | attr_to_packages.keys())

    def __getattr__(name):
        if name in subpackages:
            return importlib.import_module(f"{module_name}.{name}")
        elif name in attr_to_packages:
            subpkg = importlib.import_module(f"{package_name}.{attr_to_packages[name]}")
            return getattr(subpkg, name)
        else:
            raise AttributeError(f"No {package_name} attribute {name}")

    def __dir__():
        return __all__

    if os.environ.get("EAGER_IMPORT", ""):
        for attr in set(attr_to_packages.keys()) | subpackages:
            __getattr__(attr)

    return __getattr__, __dir__, list(__all__)


def load(fullname):
    """Return a lazily imported proxy for a module or library.

    We often see the following pattern::

      def myfunc():
          import scipy
          scipy.argmin(...)
          ....

    This is to prevent a library, in this case `scipy`, from being
    imported at function definition time, since that can be slow.

    This function provides a proxy module that, upon access, imports
    the actual module. So the idiom equivalent to the above example is::

      scipy = lazy.load("scipy")

      def myfunc():
          scipy.argmin(...)
          ....

    The initial import time is fast because the actual import is delayed
    until the first attribute is requested. The overall import time may
    decrease as well for users that don't make use of large portions
    of the library.

    Parameters
    ----------
    fullname : str
        The full name of the library to import.  For example::

          sp = lazy_import('scipy')  # import scipy as sp
          spla = lazy_import('scipy.linalg')  # import scipy.linalg as spla

    Returns
    -------
    pm : importlib.util._LazyModule
        Proxy module.  Can be used like any regularly imported module.
        Actual loading of the module occurs upon first attribute request.

    """
    try:
        return sys.modules[fullname]
    except:
        pass

    spec = importlib.util.find_spec(fullname)
    if spec is None:
        raise ModuleNotFoundError(f"No module named '{fullname}'")
    module = importlib.util.module_from_spec(spec)
    loader = importlib.util.LazyLoader(spec.loader)
    sys.modules[fullname] = module
    loader.exec_module(module)
    return module
