import pygame
from ui import draw_text
from settings import *

class Button:
    def __init__(self, x, y, width, height, text="", icon=None):
        self.rect = pygame.Rect(x, y, width, height)
        self.text = text
        self.icon = icon
        self.is_hovered = False

         # Lưu trữ icon gốc và icon đã được co giãn riêng biệt.
        self.original_icon = icon 
        self.scaled_icon = None

        # Nếu có icon được truyền vào, hãy co giãn nó ngay lập tức.
        if self.original_icon:
            # pygame.transform.smoothscale cho chất lượng hình ảnh tốt hơn so với scale thông thường.
            self.scaled_icon = pygame.transform.smoothscale(self.original_icon, (width, height))

        
        # === BIẾN MỚI CHO HIỆU ỨNG GIỮ CHỌN ===
        self.hold_start_time = 0
        self.is_holding = False
        self.hold_duration = 2.0  # Thời gian cần giữ (giây)

    def check_hover(self, pos):
        self.is_hovered = self.rect.collidepoint(pos)
        return self.is_hovered

    # === HÀM MỚI ĐỂ XỬ LÝ VIỆC GIỮ ===
    def update_hold(self, is_click_gesture):
        """
        Cập nhật trạng thái giữ của button.
        Trả về True nếu quá trình giữ hoàn tất (đủ 2 giây).
        """
        if self.is_hovered and is_click_gesture:
            if not self.is_holding:
                # Bắt đầu giữ
                self.is_holding = True
                self.hold_start_time = pygame.time.get_ticks() / 1000.0
            
            # Kiểm tra xem đã giữ đủ lâu chưa
            elapsed_time = (pygame.time.get_ticks() / 1000.0) - self.hold_start_time
            if elapsed_time >= self.hold_duration:
                self.reset_hold() # Reset lại để tránh trigger liên tục
                return True # Hoàn tất!
        else:
            # Nếu di chuyển ra ngoài hoặc nhả cử chỉ, reset lại
            self.reset_hold()
            
        return False # Chưa hoàn tất

    def reset_hold(self):
        self.is_holding = False
        self.hold_start_time = 0

    def draw(self, surface):
        color = BUTTON_HOVER_COLOR if self.is_hovered else BUTTON_COLOR
        pygame.draw.rect(surface, color, self.rect, border_radius=15)
        
        if self.text:
            draw_text(surface, self.text, self.rect.center, FONT_PATH_BOLD, 40, WHITE)
        elif self.scaled_icon:
            icon_rect = self.scaled_icon.get_rect(center=self.rect.center)
            surface.blit(self.scaled_icon, icon_rect)
            
        # === VẼ THANH TIẾN TRÌNH KHI ĐANG GIỮ ===
        if self.is_holding:
            elapsed_time = (pygame.time.get_ticks() / 1000.0) - self.hold_start_time
            progress = min(elapsed_time / self.hold_duration, 1.0) # Giá trị từ 0.0 đến 1.0
            
            progress_bar_width = self.rect.width * progress
            progress_bar_rect = pygame.Rect(self.rect.x, self.rect.y, progress_bar_width, self.rect.height)
            
            # Tạo một surface tạm để vẽ thanh tiến trình với bo góc
            progress_surface = pygame.Surface(self.rect.size, pygame.SRCALPHA)
            pygame.draw.rect(progress_surface, BUTTON_PROGRESS_COLOR, (0, 0, progress_bar_width, self.rect.height), border_radius=15)
            
            # Vẽ lên màn hình chính
            surface.blit(progress_surface, self.rect.topleft, special_flags=pygame.BLEND_RGBA_ADD)