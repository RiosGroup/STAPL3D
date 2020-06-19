#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""Tests for the stapl3d module.
"""
import pytest

from stapl3d import stapl3d


def test_something():
    assert True


def test_with_error():
    with pytest.raises(ValueError):
        # Do something that raises a ValueError
        raise(ValueError)


# Fixture example
@pytest.fixture
def an_object():
    return {}


def test_stapl3d(an_object):
    assert an_object == {}
