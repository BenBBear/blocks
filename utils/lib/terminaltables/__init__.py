"""Generate simple tables in terminals from a nested list of strings.

Use SingleTable or DoubleTable instead of AsciiTable for box-drawing characters.

https://github.com/Robpol86/terminaltables
https://pypi.python.org/pypi/terminaltables
"""

from ascii_table import AsciiTable  # noqa
from github_table import GithubFlavoredMarkdownTable  # noqa
from other_tables import DoubleTable  # noqa
from other_tables import SingleTable  # noqa
from other_tables import PorcelainTable  # noqa

__author__ = '@Robpol86'
__license__ = 'MIT'
__version__ = '3.1.0'
