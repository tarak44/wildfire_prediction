from wildfire_mlops.core import get_settings


def test_settings_defaults():
    settings = get_settings()
    assert settings.image_size == 224
    assert settings.model_path
