import os
import sys
import warnings

# Ensure project root is on sys.path for `import app`, `import api`, etc.
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain.*")

# Suppress swig-related deprecation warnings (common in some dependencies)
warnings.filterwarnings("ignore", message="builtin type .* has no __module__ attribute")
