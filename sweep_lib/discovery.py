"""Finding a sweep's run tags under runs/ and grouping them.

A sweep encodes its swept settings in the run tag, so discovery is a glob
plus a regex, and what a report plots against is some function of the match.
Reports that instead name their runs explicitly can build the same
`{key: [tag]}` mapping by hand and use it interchangeably.
"""

import glob
import os
import re
from typing import Callable, Iterator

from paths import run_dir


def matching_tags(run_glob: str, tag_re: re.Pattern) -> Iterator[tuple[str, re.Match]]:
    """(tag, match) for each run directory matching both `run_glob` and
    `tag_re`, in sorted tag order.

    The glob narrows the directory listing and the regex pins the tag's shape;
    a tag the glob catches but the regex doesn't is skipped, so a neighbouring
    sweep sharing a prefix doesn't leak in.
    """
    for path in sorted(glob.glob(run_dir(run_glob))):
        tag = os.path.basename(path)
        m = tag_re.match(tag)
        if m is not None:
            yield tag, m


def group_tags(
    run_glob: str, tag_re: re.Pattern, key: Callable[[re.Match], object | None]
) -> dict[object, list[str]]:
    """Matching tags grouped by `key(match)`.

    `key` returning None drops the tag, which is how a report excludes an arm
    it isn't ready to plot.
    """
    by_key: dict[object, list[str]] = {}
    for tag, m in matching_tags(run_glob, tag_re):
        k = key(m)
        if k is not None:
            by_key.setdefault(k, []).append(tag)
    return by_key
