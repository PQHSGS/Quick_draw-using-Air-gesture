# config_manager.py
import json

CONFIG_FILE = 'config.json'

DEFAULT_SETTINGS = {
    "game_time": 60,
    "brush_color": [255, 255, 0],
    "background_color": [0, 0, 0]
}

def load_settings():
    """Tải cài đặt từ file config.json. Nếu file không tồn tại, tạo file mới với giá trị mặc định."""
    try:
        with open(CONFIG_FILE, 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        save_settings(DEFAULT_SETTINGS)
        return DEFAULT_SETTINGS

def save_settings(settings):
    """Lưu từ điển cài đặt vào file config.json."""
    with open(CONFIG_FILE, 'w') as f:
        json.dump(settings, f, indent=4)