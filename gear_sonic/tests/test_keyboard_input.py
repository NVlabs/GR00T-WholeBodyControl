import pytest

from gear_sonic.utils.teleop.keyboard_input import KeyboardControlSource


def test_keyboard_source_projects_existing_controller_schema():
    source = KeyboardControlSource(ttl_sec=0.25)
    source.feed_key("a", now=10.0)
    source.feed_key("g", now=10.0)

    data = source.get_controller_data(now=10.1)

    assert data["right_primary_click"] == 1.0
    assert data["left_squeeze_value"] == 1.0
    assert data["right_secondary_click"] == 0.0
    assert data["left_primary_click"] == 0.0
    assert data["left_menu_button"] is False
    assert data["left_thumbstick_click"] == 0.0


def test_keyboard_source_projects_menu_and_axis_click_buttons():
    source = KeyboardControlSource(ttl_sec=0.25)
    source.feed_key("p", now=15.0)
    source.feed_key("v", now=15.0)

    data = source.get_controller_data(now=15.1)

    assert data["left_menu_button"] is True
    assert data["left_thumbstick_click"] == 1.0


def test_keyboard_source_projects_atomic_start_stop_without_partial_chord():
    source = KeyboardControlSource(ttl_sec=0.25)
    source.feed_key("s", now=20.0)

    data = source.get_controller_data(now=20.2)

    assert data["right_primary_click"] == 1.0
    assert data["right_secondary_click"] == 1.0
    assert data["left_primary_click"] == 1.0
    assert data["left_secondary_click"] == 1.0


def test_keyboard_source_projects_atomic_record_action():
    source = KeyboardControlSource(ttl_sec=0.25)
    source.feed_key("c", now=25.0)

    data = source.get_controller_data(now=25.1)

    assert data["right_primary_click"] == 1.0
    assert data["left_squeeze_value"] == 1.0
    assert data["right_secondary_click"] == 0.0


def test_keyboard_source_expires_presses_without_key_release_events():
    source = KeyboardControlSource(ttl_sec=0.25)
    source.feed_key("a", now=30.0)

    assert source.get_controller_data(now=30.249)["right_primary_click"] == 1.0
    assert source.get_controller_data(now=30.25)["right_primary_click"] == 0.0


def test_keyboard_source_rejects_ambiguous_mapping():
    with pytest.raises(ValueError, match="unique"):
        KeyboardControlSource(key_mapping={"a": "q", "b": "q"})
