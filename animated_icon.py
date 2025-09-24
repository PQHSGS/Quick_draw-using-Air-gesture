import pygame
import math
import random
from settings import WIDTH, HEIGHT

class AnimatedIcon:
    def __init__(self, icon_list):
        # Chọn ngẫu nhiên một icon từ danh sách được cung cấp
        self.original_image = random.choice(icon_list)
        # Co giãn icon về một kích thước hợp lý để hiển thị
        self.original_image = pygame.transform.smoothscale(self.original_image, (300, 300))
        
        self.image = self.original_image.copy()
        self.rect = self.image.get_rect(center=(WIDTH / 2, HEIGHT / 2))
        
        # Thuộc tính cho animation
        self.start_time = pygame.time.get_ticks()
        self.duration = 2000  # Animation kéo dài 2 giây (2000 ms)
        self.sway_angle = 15  # Góc lắc lư tối đa (độ)
        self.sway_speed = 3   # Tốc độ lắc lư
        
        self.is_finished = False

    def update(self):
        # Tính toán thời gian đã trôi qua
        elapsed_time = pygame.time.get_ticks() - self.start_time
        
        if elapsed_time > self.duration:
            self.is_finished = True
            return

        # --- Logic Lắc Lư (Sway) ---
        # Sử dụng hàm sin để tạo ra chuyển động qua lại mượt mà
        angle = self.sway_angle * math.sin(elapsed_time / 1000.0 * self.sway_speed * math.pi)
        
        # Xoay ảnh gốc (để tránh mất chất lượng ảnh sau mỗi lần xoay)
        self.image = pygame.transform.rotozoom(self.original_image, angle, 1.0)
        self.rect = self.image.get_rect(center=(WIDTH / 2, HEIGHT / 2))
        
        # --- Logic Mờ Dần (Fade Out) ---
        # Bắt đầu mờ dần trong 0.5 giây cuối cùng
        if elapsed_time > self.duration - 500:
            fade_progress = (elapsed_time - (self.duration - 500)) / 500.0
            alpha = 255 * (1 - fade_progress)
            self.image.set_alpha(alpha)

    def draw(self, surface):
        if not self.is_finished:
            surface.blit(self.image, self.rect)


# === LỚP MỚI ĐỂ XỬ LÝ ICON TĨNH ===
class StaticIcon:
    """
    Lớp này "bọc" một hình ảnh tĩnh để nó có các phương thức giống hệt AnimatedIcon,
    giúp mã nguồn trong PlayingState trở nên đồng nhất.
    """
    def __init__(self, image):
        # Co giãn ảnh về kích thước cố định để hiển thị nhất quán
        self.image = pygame.transform.smoothscale(image, (150, 150))
        self.rect = self.image.get_rect(center=(WIDTH / 2, HEIGHT / 2))
        
        # Thiết lập thời gian hiển thị
        self.start_time = pygame.time.get_ticks()
        self.duration = 1500 # Hiển thị trong 1.5 giây
        self.is_finished = False

    def update(self):
        """
        Phương thức update của icon tĩnh chỉ kiểm tra xem đã hết thời gian hiển thị chưa.
        """
        elapsed_time = pygame.time.get_ticks() - self.start_time
        if elapsed_time > self.duration:
            self.is_finished = True

    def draw(self, surface):
        """
        Phương thức draw chỉ đơn giản là vẽ hình ảnh lên màn hình.
        """
        if not self.is_finished:
            surface.blit(self.image, self.rect)