# ModelArchitect.py

# Mô tả: Định nghĩa kiến trúc của mô hình mạng neural tích chập (CNN) nhỏ,
#        dựa trên kiến trúc VGG, sử dụng thư viện PyTorch.
#        Mô hình này được thiết kế để phân loại các hình ảnh vẽ tay có kích thước nhỏ.

import torch
import torch.nn as nn

class VGG_Small(nn.Module):
    """
    Một phiên bản rút gọn của kiến trúc VGG (Visual Geometry Group).
    Mô hình này bao gồm 3 khối tích chập (convolutional blocks) để trích xuất đặc trưng
    và một bộ phân loại (classifier) gồm các lớp kết nối đầy đủ (fully connected layers)
    để đưa ra dự đoán cuối cùng.
    """
    def __init__(self, num_classes):
        """
        Hàm khởi tạo, nơi các lớp (layers) của mạng neural được định nghĩa.

        Args:
            num_classes (int): Số lượng lớp đầu ra, tương ứng với số lượng
                               đối tượng mà mô hình cần phân loại (ví dụ: 'táo', 'chuối',...).
        """
        # Gọi hàm khởi tạo của lớp cha (nn.Module)
        super().__init__()

        # ----------------------------------------------------------------------
        # Phần 1: Khối trích xuất đặc trưng (Feature Extractor)
        # ----------------------------------------------------------------------
        # Sử dụng nn.Sequential để nhóm các lớp lại với nhau. Dữ liệu sẽ đi qua
        # từng lớp theo thứ tự đã định nghĩa.
        self.features = nn.Sequential(
            # --- Khối 1 ---
            # Đầu vào: ảnh xám 1 kênh, kích thước (batch, 1, 32, 32)
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding=1),
            # Lớp tích chập đầu tiên:
            # - in_channels=1: nhận ảnh đầu vào là ảnh xám (1 kênh màu).
            # - out_channels=64: tạo ra 64 bản đồ đặc trưng (feature maps).
            # - kernel_size=3: kích thước của bộ lọc là 3x3.
            # - padding=1: thêm 1 lớp pixel 0 xung quanh ảnh để kích thước không gian
            #              (chiều rộng, chiều cao) không đổi sau phép tích chập.
            nn.ReLU(inplace=True), # Hàm kích hoạt phi tuyến tính, giúp mô hình học các mối quan hệ phức tạp.
                                   # inplace=True giúp tiết kiệm bộ nhớ.
            nn.Conv2d(64, 64, kernel_size=3, padding=1), # Lớp tích chập thứ hai.
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # Lớp gộp cực đại (Max Pooling) với cửa sổ 2x2.
                             # Giảm kích thước không gian của feature map đi một nửa (từ 32x32 -> 16x16)
                             # và giữ lại các đặc trưng quan trọng nhất.

            # --- Khối 2 ---
            # Đầu vào từ khối 1: (batch, 64, 16, 16)
            nn.Conv2d(64, 128, 3, padding=1), # Tăng số kênh lên 128.
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # Giảm kích thước từ 16x16 -> 8x8.

            # --- Khối 3 ---
            # Đầu vào từ khối 2: (batch, 128, 8, 8)
            nn.Conv2d(128, 256, 3, padding=1), # Tăng số kênh lên 256.
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2), # Giảm kích thước từ 8x8 -> 4x4.
            # Đầu ra của self.features sẽ có kích thước: (batch, 256, 4, 4)
        )

        # ----------------------------------------------------------------------
        # Phần 2: Bộ phân loại (Classifier)
        # ----------------------------------------------------------------------
        # Phần này nhận đầu ra đã được trích xuất đặc trưng và đưa ra dự đoán.
        self.classifier = nn.Sequential(
            nn.Flatten(), # "Làm phẳng" đầu ra từ khối features.
                          # Chuyển tensor 4D (batch, 256, 4, 4) thành 2D (batch, 256*4*4).
                          # Kích thước sau khi làm phẳng: (batch, 4096).

            nn.Linear(256*4*4, 512), # Lớp kết nối đầy đủ (Fully Connected Layer).
                                     # Nhận đầu vào 4096 features và biến đổi thành 512 features.
            nn.ReLU(inplace=True),

            nn.Dropout(0.5), # Lớp Dropout: "tắt" ngẫu nhiên 50% các neuron trong quá trình huấn luyện.
                             # Đây là một kỹ thuật điều chuẩn (regularization) để chống lại hiện tượng
                             # học vẹt (overfitting).

            nn.Linear(512, num_classes) # Lớp kết nối đầy đủ cuối cùng.
                                        # Ánh xạ 512 features thành `num_classes` điểm số (logits),
                                        # mỗi điểm số tương ứng với một lớp đối tượng.
        )

    def forward(self, x):
        """
        Định nghĩa luồng đi của dữ liệu qua mạng (forward pass).

        Args:
            x (torch.Tensor): Tensor đầu vào, thường là một batch các hình ảnh.
                              Kích thước mong đợi: (batch_size, 1, 32, 32).

        Returns:
            torch.Tensor: Tensor chứa các điểm số (logits) cho mỗi lớp.
                          Kích thước: (batch_size, num_classes).
        """
        # 1. Cho dữ liệu đi qua khối trích xuất đặc trưng.
        x = self.features(x)
        # 2. Cho kết quả đi qua bộ phân loại.
        x = self.classifier(x)
        # 3. Trả về kết quả đầu ra (logits).
        return x