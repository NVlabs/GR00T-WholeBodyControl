#!/usr/bin/env python3
"""Compatibility entrypoint for schema-2 causal gantry authorization."""

from gear_sonic.scripts.authorize_g1_true23_causal_gantry import (
    AUTHORIZATION_PHRASE,
    KIND,
    SCHEMA_VERSION,
    _body,
    _parser,
    main,
    validate_causal_live_shadow_evidence,
)

__all__ = [
    "AUTHORIZATION_PHRASE",
    "KIND",
    "SCHEMA_VERSION",
    "_body",
    "_parser",
    "validate_causal_live_shadow_evidence",
    "main",
]


if __name__ == "__main__":
    raise SystemExit(main())
