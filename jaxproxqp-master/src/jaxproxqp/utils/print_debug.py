import functools as ft
import sys

import jax.debug as jd


def _format_print_callback(fmt: str, *args, **kwargs):
    sys.stdout.write(fmt.format(*args, **kwargs) + "\n")


def print(fmt: str, *args, ordered: bool = True, **kwargs):
    jd.callback(ft.partial(_format_print_callback, fmt), *args, **kwargs, ordered=ordered)
