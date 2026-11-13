"""Canonical identities and strict JSON decoding, without logging input data."""

import hashlib
import json
import math
import re
from pathlib import Path

from .errors import ControlError

IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,95}\Z")
SHA256 = re.compile(r"[0-9a-f]{64}\Z")


def identifier(value):
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ControlError("INVALID_IDENTIFIER")
    return value


def sha256(value):
    if not isinstance(value, str) or not SHA256.fullmatch(value):
        raise ControlError("INVALID_SHA256")
    return value


def canonical(value):
    try:
        return json.dumps(value, sort_keys=True, ensure_ascii=True,
                          separators=(",", ":"), allow_nan=False)
    except (TypeError, ValueError):
        raise ControlError("INVALID_JSON_VALUE") from None


def digest(value):
    return hashlib.sha256(canonical(value).encode("utf-8")).hexdigest()


def _pairs(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ControlError("DUPLICATE_JSON_KEY")
        result[key] = value
    return result


def _constant(_):
    raise ControlError("NONFINITE_JSON_NUMBER")


def _float(value):
    number = float(value)
    if not math.isfinite(number):
        raise ControlError("NONFINITE_JSON_NUMBER")
    return number


def parse_json(data):
    try:
        return json.loads(data, object_pairs_hook=_pairs, parse_constant=_constant, parse_float=_float)
    except (UnicodeError, ValueError, RecursionError):
        raise ControlError("JSON_READ_FAILED") from None


def read_json(path, *, max_bytes=1_048_576):
    if type(max_bytes) is not int or not 1<=max_bytes<=16*1024*1024:
        raise ControlError('INVALID_JSON_SIZE_LIMIT')
    try:
        with Path(path).open("rb") as stream:
            data = stream.read(max_bytes+1)
        if len(data) > max_bytes:
            raise ControlError("JSON_TOO_LARGE")
        return parse_json(data.decode("utf-8"))
    except (OSError, UnicodeError, ValueError, RecursionError):
        raise ControlError("JSON_READ_FAILED") from None
