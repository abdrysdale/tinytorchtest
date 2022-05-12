"""Project version"""

# Python imports
import toml

# Version
__version__ = toml.load("pyproject.toml")["tool"]["poetry"]["version"]
