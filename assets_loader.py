# assets_loader.py
import pygame
from settings import ASSETS_PATH

def load_animated_icons(path, class_names_vn):
    """
    Tải tất cả các icon từ cấu trúc thư mục mới (icon_v2).
    Trả về một dictionary với key là class_id và value là một danh sách các icon (pygame.Surface).
    """
    animated_icons = {}
    icon_folder = Path(path)
    if not icon_folder.exists():
        print(f"Cảnh báo: Thư mục icon động không tồn tại tại {icon_folder}")
        return animated_icons

    # Tạo một bản đồ từ tên tiếng Việt sang tên thư mục (tiếng Anh, không dấu)
    # Ví dụ: 'Quả táo' -> 'apple'. Bạn cần tùy chỉnh cho khớp với tên thư mục của bạn.
    name_map = {
        'Quả táo': 'apple', 'Quả chuối': 'banana', 'Bánh trung thu': 'cake',
        'Con tàu': 'cruise_ship', 'Mặt nạ': 'face', 'Bánh cá': 'fish',
        'Bông hoa': 'flower', 'Đèn lồng': 'lantern', 'Con lân': 'lion',
        'Ông trăng': 'moon', 'Quả lê': 'pear', 'Quả dứa': 'pineapple',
        'Thỏ ngọc': 'rabbit', 'Đèn ông sao': 'star', 'Quả dâu tây': 'strawberry',
        'Cây thần': 'tree','Dưa hấu': 'watermelon'
        # Thêm các ánh xạ khác nếu cần
    }

    # Tạo bản đồ ngược từ tên thư mục sang class_id
    folder_to_id = {}
    for i, class_name_vn in enumerate(class_names_vn):
        if class_name_vn in name_map:
            folder_name = name_map[class_name_vn]
            folder_to_id[folder_name] = i

    # Duyệt qua các thư mục con trong icon_v2
    for class_dir in icon_folder.iterdir():
        if not class_dir.is_dir():
            continue
        
        folder_name = class_dir.name
        if folder_name in folder_to_id:
            class_id = folder_to_id[folder_name]
            animated_icons[class_id] = []
            
            # Tải tất cả các ảnh trong thư mục con này
            for img_file in sorted(class_dir.glob("*.png")):
                image = pygame.image.load(img_file).convert_alpha()
                animated_icons[class_id].append(image)

    print(f"Đã tải thành công {sum(len(v) for v in animated_icons.values())} icon động cho {len(animated_icons)} lớp.")
    return animated_icons


def load_all_assets():
    """Tải tất cả hình ảnh và âm thanh cần thiết cho game."""
    assets = {
        'icons': {},
        'sounds': {}
    }
    
    # Tải icons
    try:
        correct_icon = pygame.image.load(ASSETS_PATH / "icons" / "correct.png").convert_alpha()
        incorrect_icon = pygame.image.load(ASSETS_PATH / "icons" / "incorrect.png").convert_alpha()
        assets['icons']['correct'] = pygame.transform.scale(correct_icon, (200, 200))
        assets['icons']['incorrect'] = pygame.transform.scale(incorrect_icon, (200, 200))
        assets['icons']['pause'] = pygame.image.load(ASSETS_PATH / "icons" / "pause_icon.png").convert_alpha()
        assets['icons']['play'] = pygame.image.load(ASSETS_PATH / "icons" / "play_icon.png").convert_alpha()
        assets['icons']['home'] = pygame.image.load(ASSETS_PATH / "icons" / "home_icon.png").convert_alpha()
        print("Tải icons thành công.")
    except Exception as e:
        print(f"Lỗi khi tải icons: {e}")

    # Tải âm thanh
    try:
        pygame.mixer.init() # Khởi tạo module âm thanh
        assets['sounds']['correct'] = pygame.mixer.Sound(ASSETS_PATH / "sounds" / "correct.wav")
        assets['sounds']['incorrect'] = pygame.mixer.Sound(ASSETS_PATH / "sounds" / "incorrect.wav")
        print("Tải âm thanh thành công.")
    except Exception as e:
        print(f"Lỗi khi tải âm thanh: {e}")

    return assets