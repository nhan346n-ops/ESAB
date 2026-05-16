"""Functions helping to work with filenames"""

import glob
from fnmatch import fnmatch
from pathlib import Path

import os
import itertools
import pathlib
from typing import List, Tuple, Set, Iterable, Optional


def splitext_of_fname(fname: str) -> Tuple[str, str]:
    """
    >>> splitext_of_fname('./a/data.xsf')
    ('./a/data', 'xsf')
    >>> splitext_of_fname('./a/data.xsf.nc')
    ('./a/data', 'xsf.nc')
    """
    fname, last_ext = os.path.splitext(fname)
    exts = [last_ext]
    while last_ext:
        fname, last_ext = os.path.splitext(fname)
        exts.append(last_ext)
    ext = "".join(reversed(exts[:-1])).lstrip(".")
    return fname, ext


def basename_of_fname(fname: str) -> str:
    return os.path.split(splitext_of_fname(fname)[0])[1]


def ext_of_fname(fname: str) -> str:
    return splitext_of_fname(fname)[1]


def replace_path_prefix(path: str, prefix: str, new: str) -> str:
    """
    >>> replace_path_prefix('/run/data', '/run', '/tmp')
    '/tmp/data'
    >>> replace_path_prefix('run/data/last/dir', 'run/data/', 'tmp')
    'tmp/last/dir'
    """
    if not path.startswith(prefix):
        raise ValueError(f"Given {path} doesn't start with {prefix}.")
    return os.path.join(new, path[len(prefix) :].lstrip(r"\/"))


def scan_dir(dirname, file_pattern: Iterable[str] | None = None, recurse=False):
    """Scan and Recurse through a directory and return an iterator matching one of the pattern defined"""
    try:
        if file_pattern is None:
            file_pattern = ["*.*"]
        with os.scandir(dirname) as it:
            for entry in it:
                try:
                    if entry.is_dir() and recurse:
                        yield from scan_dir(entry.path, file_pattern, recurse)
                    if entry.is_file() and any(fnmatch(entry.name, pattern) for pattern in file_pattern):
                        yield Path(entry.path)
                except OSError:
                    pass
    except OSError:
        pass


# pylint:disable = superfluous-parens,dangerous-default-value
def repr_file_tree(
    tree: dict, root: Optional[str] = None, margin: str = "  ", *, indent: int = 0, continued_levels: Set[int] = set()
) -> Iterable[str]:
    """Yield lines representing given tree graphically"""
    if root is None:  # no root given, let's find them and act on each
        all_succs = set(itertools.chain.from_iterable(tree.values()))
        roots = tuple(item for item in tree if item not in all_succs)
        if len(roots) >= 1:
            yield "[ROOT]"
            *roots, last_root = sorted(roots)
            for found in roots:
                yield from repr_file_tree(
                    tree, found, margin, indent=indent + 1, continued_levels=continued_levels | {indent}
                )
            yield from repr_file_tree(tree, last_root, margin, indent=indent + 1, continued_levels=continued_levels)
    else:  # root was given ; lets yield its representation
        yield "".join(
            margin + ("┃   " if tab in continued_levels else "    ") for tab in range(indent - 1)
        ) + margin + (("┠──►" if (indent - 1) in continued_levels else "┖──►") if indent >= 1 else "") + root
        succs = tuple(tree.get(root, ()))
        if len(succs) > 0:
            *succs, last = sorted(succs)
            for succ in succs:
                yield from repr_file_tree(
                    tree, succ, margin=margin, indent=indent + 1, continued_levels=continued_levels | {indent}
                )
            yield from repr_file_tree(tree, last, margin=margin, indent=indent + 1, continued_levels=continued_levels)


def delete_files(with_pattern: str):
    """
    Delete all files with the same basename pattern (i.e. /foo/bar/basename.*) in the same directory.
    Supply the file path pattern : with_pattern = "/foo/bar/basename"
    Use it to completely remove shapefiles and their associated files (such as .shp, .shx, .dbf, etc.)
    """
    try:
        to_delete = glob.glob(with_pattern + ".*")
        for file in to_delete:
            os.remove(file)
    except Exception as e:
        print(e)


def read_filelist(filepath: str | os.PathLike) -> List[str]:
    """
    Read a text file containing file list
    """
    with open(file=filepath, encoding="utf-8") as file:
        return file.read().splitlines()
