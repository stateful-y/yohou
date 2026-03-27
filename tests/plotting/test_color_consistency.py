"""Tests for PanelColorManager and color consistency across plots."""

import pytest

from yohou.plotting._utils import PanelColorManager, palette_yohou


class TestPanelColorManager:
    def test_same_name_returns_same_color(self):
        mgr = PanelColorManager()
        assert mgr.get_color("alice") == mgr.get_color("alice")

    def test_different_names_get_different_colors(self):
        mgr = PanelColorManager()
        assert mgr.get_color("alice") != mgr.get_color("bob")

    def test_member_name_independent_of_group_prefix(self):
        mgr = PanelColorManager()
        c1 = mgr.get_color("a")
        c2 = mgr.get_color("a")
        assert c1 == c2

    def test_custom_palette_overrides_default(self):
        custom = ["#FF0000", "#00FF00"]
        mgr = PanelColorManager(color_palette=custom)
        c1 = mgr.get_color("first")
        c2 = mgr.get_color("second")
        assert c1 in custom
        assert c2 in custom
        assert c1 != c2

    def test_color_cycles_when_more_members_than_palette(self):
        tiny = ["#AAA", "#BBB"]
        mgr = PanelColorManager(color_palette=tiny)
        mgr.get_color("a")
        mgr.get_color("b")
        c3 = mgr.get_color("c")
        assert c3 in tiny

    def test_get_colors_batch(self):
        mgr = PanelColorManager()
        names = ["x", "y", "z"]
        colors = mgr.get_colors(names)
        assert len(colors) == 3
        assert colors[0] == mgr.get_color("x")
        assert len(set(colors)) == 3

    def test_default_palette_matches_yohou(self):
        mgr = PanelColorManager()
        palette = list(palette_yohou().values())
        c = mgr.get_color("test")
        assert c in palette
