"""Utility functions: mpf JSON serialization, data save/load."""

import json
import os
from mpmath import mp, mpf


class MpfEncoder(json.JSONEncoder):
    """JSON encoder that serializes mpf as strings."""
    def default(self, obj):
        if isinstance(obj, mpf):
            return {"__mpf__": mp.nstr(obj, mp.dps)}
        return super().default(obj)


def mpf_decoder(dct):
    """JSON object hook to deserialize mpf values."""
    if "__mpf__" in dct:
        return mpf(dct["__mpf__"])
    return dct


def save_coefficients(coeffs, filepath, metadata=None):
    """Save a list of mpf coefficients to JSON."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    data = {
        "dps": mp.dps,
        "count": len(coeffs),
        "coefficients": coeffs,
    }
    if metadata:
        data["metadata"] = metadata
    with open(filepath, "w") as f:
        json.dump(data, f, cls=MpfEncoder, indent=2)


def load_coefficients(filepath):
    """Load coefficients from JSON. Returns (coeffs, metadata)."""
    with open(filepath, "r") as f:
        data = json.load(f, object_hook=mpf_decoder)
    coeffs = data["coefficients"]
    metadata = data.get("metadata", {})
    return coeffs, metadata


def save_text(text, filepath):
    """Save plain text to file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "w") as f:
        f.write(text)
