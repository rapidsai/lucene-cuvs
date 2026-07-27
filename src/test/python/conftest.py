# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import os

import pytest


CASE_MARKER = "pylucene_case"
CASE_ENVIRONMENT_VARIABLE = "CUVS_LUCENE_PYLUCENE_CASES"
DEFAULT_CASE_SELECTION = "cpu-hnsw-single-document-index"


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        f"{CASE_MARKER}(*selectors): select a PyLucene end-to-end scenario",
    )


def _requested_selectors() -> tuple[frozenset[str], bool]:
    configured_value = os.environ.get(CASE_ENVIRONMENT_VARIABLE)
    configured = configured_value or DEFAULT_CASE_SELECTION
    selectors = frozenset(
        selector.strip()
        for selector in configured.split(",")
        if selector.strip()
    )
    if not selectors:
        raise pytest.UsageError(
            f"{CASE_ENVIRONMENT_VARIABLE} did not select any cases"
        )
    return selectors, configured_value is not None


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    requested, explicitly_configured = _requested_selectors()
    selectable_items: list[tuple[pytest.Item, frozenset[str]]] = []
    available: set[str] = set()

    for item in items:
        marker = item.get_closest_marker(CASE_MARKER)
        if marker is None:
            continue
        selectors = frozenset(str(value) for value in marker.args)
        selectable_items.append((item, selectors))
        available.update(selectors)

    if not selectable_items:
        return

    unknown = requested - available
    if unknown and not explicitly_configured:
        return
    if unknown:
        raise pytest.UsageError(
            "Unknown PyLucene case or group: "
            + ", ".join(sorted(unknown))
        )

    deselected = [
        item
        for item, selectors in selectable_items
        if requested.isdisjoint(selectors)
    ]
    if deselected:
        config.hook.pytest_deselected(items=deselected)
        deselected_set = set(deselected)
        items[:] = [item for item in items if item not in deselected_set]
