"""Pure reservation arithmetic. Durable transactions are owned by RunStore."""

from .errors import ControlError

RESOURCES = {
    "editor_requests": "subscription_editor_request_limit",
    "dev_requests": "dev_forward_request_limit", "dev_items": "dev_forward_item_limit",
    "baseline_requests": "baseline_forward_request_limit", "baseline_items": "baseline_forward_item_limit",
    "evaluation_requests": "evaluation_forward_request_limit", "evaluation_items": "evaluation_forward_item_limit",
    "generated_tokens": "generated_token_limit", "activation_requests": "activation_probe_limit",
    "activation_bytes": "activation_bytes_limit",
}
REQUEST_RESOURCE = {"editor": "editor_requests", "dev_forward": "dev_requests",
                    "baseline": "baseline_requests", "evaluation": "evaluation_requests",
                    "activation": "activation_requests"}


def reservation(kind, *, items=0, max_new_tokens=0, activation_bytes=0):
    if kind not in REQUEST_RESOURCE:
        raise ControlError("UNKNOWN_REQUEST_KIND")
    if any(type(n) is not int or n < 0 for n in (items, max_new_tokens, activation_bytes)):
        raise ControlError("INVALID_REQUEST_BOUNDS")
    cost = {REQUEST_RESOURCE[kind]: 1}
    if kind in ("dev_forward", "baseline", "evaluation"):
        if items < 1 or max_new_tokens < 1 or activation_bytes:
            raise ControlError("INVALID_REQUEST_BOUNDS")
        item_key = {"dev_forward": "dev_items", "baseline": "baseline_items", "evaluation": "evaluation_items"}[kind]
        cost.update({item_key: items, "generated_tokens": items * max_new_tokens})
    elif kind == "activation":
        if activation_bytes < 1 or items or max_new_tokens:
            raise ControlError("INVALID_REQUEST_BOUNDS")
        cost["activation_bytes"] = activation_bytes
    elif items or max_new_tokens or activation_bytes:
        raise ControlError("INVALID_REQUEST_BOUNDS")
    return cost


def actual_usage(kind, reserved, supplied):
    """None conservatively charges the entire bound, including failed calls."""
    if supplied is None:
        return dict(reserved)
    if not isinstance(supplied, dict) or set(supplied) != set(reserved):
        raise ControlError("INVALID_USAGE")
    if any(type(value) is not int or value < 0 or value > reserved[key] for key, value in supplied.items()):
        raise ControlError("USAGE_EXCEEDS_RESERVATION")
    if supplied[REQUEST_RESOURCE[kind]] != 1:
        raise ControlError("REQUEST_MUST_BE_CHARGED")
    return supplied
