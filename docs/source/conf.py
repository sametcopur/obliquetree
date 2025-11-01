import os
import sys
import types
from typing import TypeVar, Generic
import numpy as np
from typing import TYPE_CHECKING

# Add parent directory to path
sys.path.insert(0, os.path.abspath("../.."))

# Mock modules setup
mock_base_module = types.ModuleType("obliquetree.src.base")

class TreeClassifier:
    pass

mock_base_module.TreeClassifier = TreeClassifier
sys.modules["obliquetree.src.base"] = mock_base_module

# Utils mock
mock_utils_module = types.ModuleType("obliquetree.src.utils")
def mock_export_tree(*args, **kwargs): pass
mock_utils_module.export_tree = mock_export_tree
sys.modules["obliquetree.src.utils"] = mock_utils_module

# Project information
project = "obliquetree"
copyright = "2025, Samet Çopur"
author = "Samet Çopur"
version = "1.0.4"
release = "1.0.4"

# Extensions configuration
extensions = [
    "sphinx.ext.autodoc",
    "sphinx.ext.autosummary",
    "sphinx.ext.napoleon",
    "sphinx_autodoc_typehints",
    "sphinx_rtd_theme",
    "sphinx.ext.mathjax",
    "myst_parser",
]

# MyST configuration
myst_enable_extensions = [
    "dollarmath",
    "amsmath",
]

# Napoleon settings
napoleon_use_param = True
napoleon_use_ivar = True
napoleon_use_rtype = True
napoleon_preprocess_types = True
napoleon_type_aliases = {
    "ArrayLike": "numpy.typing.ArrayLike",
    "NDArray": "numpy.ndarray",
}

# Type hints settings
autodoc_typehints = "description"
autodoc_typehints_format = "short"
always_document_param_types = True
typehints_fully_qualified = False
typehints_document_rtype = True

# Additional settings for numpy docstring format
napoleon_numpy_docstring = True
napoleon_google_docstring = False
napoleon_include_init_with_doc = False
napoleon_include_private_with_doc = False
napoleon_include_special_with_doc = False
napoleon_use_admonition_for_examples = False
napoleon_use_admonition_for_notes = False
napoleon_use_admonition_for_references = False
napoleon_use_keyword = True

# Theme configuration
templates_path = ["_templates"]
html_theme = "furo"
html_static_path = ["_static"]