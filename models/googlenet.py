import torch
import torch.nn as nn

# =============================================
# Inception Module - GoogLeNet의 핵심 구조
# 여러 크기의 필터를 동시에 적용하고 결과를 합침
# =============================================
class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_1x1, out_3x3_reduce, out_3x3,
                 out_5x5_reduce, out_5x5, out_pool):
        super(InceptionModule, self).__init__()

        # 1×1 conv 브랜치
        self.branch1 = nn.Sequential(
            nn.Conv2d(in_channels, out_1x1, kernel_size=1),
            nn.BatchNorm2d(out_1x1),
            nn.ReLU(inplace=True)
        )

        # 1×1 conv → 3×3 conv 브랜치
        self.branch2 = nn.Sequential(
            nn.Conv2d(in_channels, out_3x3_reduce, kernel_size=1),
            nn.BatchNorm2d(out_3x3_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_3x3_reduce, out_3x3, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_3x3),
            nn.ReLU(inplace=True)
        )

        # 1×1 conv → 5×5 conv 브랜치
        self.branch3 = nn.Sequential(
            nn.Conv2d(in_channels, out_5x5_reduce, kernel_size=1),
            nn.BatchNorm2d(out_5x5_reduce),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_5x5_reduce, out_5x5, kernel_size=5, padding=2),
            nn.BatchNorm2d(out_5x5),
            nn.ReLU(inplace=True)
        )

        # 3×3 MaxPool → 1×1 conv 브랜치
        self.branch4 = nn.Sequential(
            nn.MaxPool2d(kernel_size=3, stride=1, padding=1),
            nn.Conv2d(in_channels, out_pool, kernel_size=1),
            nn.BatchNorm2d(out_pool),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        b4 = self.branch4(x)
        # 4개 브랜치 결과를 채널 방향으로 합침
        return torch.cat([b1, b2, b3, b4], dim=1)


# =============================================
# GoogLeNet - CIFAR-10 용으로 축소된 버전
# 원본은 ImageNet(224×224)용이라
# CIFAR-10(32×32)에 맞게 구조 조정
# =============================================
class GoogLeNet(nn.Module):
    def __init__(self, num_classes=10):
        super(GoogLeNet, self).__init__()

        # 초기 특징 추출
        self.pre_layers = nn.Sequential(
            nn.Conv2d(3, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True)
        )

        # Inception 모듈 쌓기
        # InceptionModule(in_ch, 1x1, 3x3r, 3x3, 5x5r, 5x5, pool)
        self.inception3a = InceptionModule(192,  64,  96, 128, 16, 32, 32)
        self.inception3b = InceptionModule(256, 128, 128, 192, 32, 96, 64)

        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.inception4a = InceptionModule(480, 192,  96, 208, 16,  48,  64)
        self.inception4b = InceptionModule(512, 160, 112, 224, 24,  64,  64)
        self.inception4c = InceptionModule(512, 128, 128, 256, 24,  64,  64)
        self.inception4d = InceptionModule(512, 112, 144, 288, 32,  64,  64)
        self.inception4e = InceptionModule(528, 256, 160, 320, 32, 128, 128)

        self.inception5a = InceptionModule(832, 256, 160, 320, 32, 128, 128)
        self.inception5b = InceptionModule(832, 384, 192, 384, 48, 128, 128)

        # 최종 분류
        self.avgpool    = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout    = nn.Dropout(0.4)
        self.classifier = nn.Linear(1024, num_classes)

    def forward(self, x):
        x = self.pre_layers(x)

        x = self.inception3a(x)
        x = self.inception3b(x)
        x = self.maxpool(x)

        x = self.inception4a(x)
        x = self.inception4b(x)
        x = self.inception4c(x)
        x = self.inception4d(x)
        x = self.inception4e(x)
        x = self.maxpool(x)

        x = self.inception5a(x)
        x = self.inception5b(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.classifier(x)
        return x