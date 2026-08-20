"""Tiny helpers shared by every course part.

A "cell" is just a (kind, source) tuple where kind is "markdown" or "code".
Both the .ipynb builder and the PDF builder consume the same list, which is
what guarantees the notebook and the PDF can never drift apart.
"""
from textwrap import dedent


def md(text: str) -> tuple[str, str]:
    return ("markdown", dedent(text).strip("\n"))


def code(text: str) -> tuple[str, str]:
    return ("code", dedent(text).strip("\n"))
