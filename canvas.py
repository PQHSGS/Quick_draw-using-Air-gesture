# canvas.py
import pygame
from settings import *

class Canvas:
    def __init__(self, x, y, width, height):
        # rect để xác định vị trí và kích thước của canvas trên màn hình chính
        self.rect = pygame.Rect(x, y, width, height)
        # surface là nơi thực sự diễn ra việc vẽ
        self.surface = pygame.Surface((width, height))
        self.surface.fill(BLACK) # Nền đen cho canvas

        # Biến để theo dõi trạng thái vẽ
        self.is_drawing = False
        self.last_pos = None

    def start_drawing(self, pos):
        """Đánh dấu bắt đầu một nét vẽ mới."""
        self.is_drawing = True
        self.last_pos = pos

    def stop_drawing(self):
        """Kết thúc nét vẽ."""
        self.is_drawing = False
        self.last_pos = None

    def draw_line(self, pos, brush_size, color):
        """
        Vẽ lên canvas. Nối điểm hiện tại với điểm trước đó để tạo nét vẽ liền mạch.
        """
        if self.last_pos:
            # Vẽ đường thẳng từ điểm cuối cùng đến điểm hiện tại
            pygame.draw.line(self.surface, color, self.last_pos, pos, brush_size * 2)
        
        # Vẽ một vòng tròn tại điểm hiện tại để làm mịn các góc
        pygame.draw.circle(self.surface, color, pos, brush_size)
        
        # Cập nhật điểm cuối cùng
        self.last_pos = pos

    def clear(self):
        """Xóa sạch toàn bộ canvas."""
        self.surface.fill(BLACK)
        print("Canvas đã được xóa.")

    def get_surface(self):
        """Trả về surface chứa hình vẽ để có thể xử lý (ví dụ: đưa vào model AI)."""
        return self.surface

    def draw_to_screen(self, screen):
        """Hiển thị canvas và khung viền của nó lên màn hình chính."""
        # Vẽ canvas
        screen.blit(self.surface, self.rect.topleft)
        # Vẽ khung viền để người dùng biết đâu là vùng vẽ
        pygame.draw.rect(screen, GRAY, self.rect, 5) # Độ dày 5 pixel