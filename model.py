import torch
import torch.nn as nn

# ----------------------------------------------------
# 2.1. 기본 구성 블록 (Basic Building Block)
# ----------------------------------------------------
class ConvBlock(nn.Module):
    """(Conv2D -> BatchNorm -> ReLU) * 2"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv_block(x)

# ----------------------------------------------------
# 2.2. TriNet 메인 아키텍처
# ----------------------------------------------------
class TriNet(nn.Module):
    def __init__(self, in_channels=1, num_classes=11, deep_supervision=False):
        super().__init__()
        self.deep_supervision = deep_supervision
        filters = [64, 128, 256, 512, 1024] # 필터 채널 수

        # --- 공유 인코더 (Shared Encoder) ---
        self.pool = nn.MaxPool2d(2, 2)
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)

        # X_0,0 to X_3,0 (Encoder Path)
        self.conv0_0 = ConvBlock(in_channels, filters[0])
        self.conv1_0 = ConvBlock(filters[0], filters[1])
        self.conv2_0 = ConvBlock(filters[1], filters[2])
        self.conv3_0 = ConvBlock(filters[2], filters[3])

        # --- UNet++ 중간 노드 (Nested Skip-Path) ---
        # 논문에서 설명된 UNet++의 조밀한 스킵 연결 구현 [cite: 71, 101]
        self.conv0_1 = ConvBlock(filters[0] + filters[1], filters[0])
        self.conv1_1 = ConvBlock(filters[1] + filters[2], filters[1])
        self.conv2_1 = ConvBlock(filters[2] + filters[3], filters[2])

        self.conv0_2 = ConvBlock(filters[0]*2 + filters[1], filters[0])
        self.conv1_2 = ConvBlock(filters[1]*2 + filters[2], filters[1])

        self.conv0_3 = ConvBlock(filters[0]*3 + filters[1], filters[0])

        # --- 이중 디코더 (Dual Decoders) ---
        # 두 작업은 상호 보완 관계에 있음 [cite: 78]
        # 디코더 1: 노이즈 제거 (Denoising) - 회귀 문제 [cite: 96]
        self.decoder1 = nn.Sequential(
            ConvBlock(filters[0]*4, filters[0]),
            nn.Conv2d(filters[0], 1, kernel_size=1),
            nn.Tanh() # 출력을 [-1, 1] 범위로
        )

        # 디코더 2: 위상 언래핑 (Unwrapping) - 분할 문제 [cite: 100]
        self.decoder2 = nn.Sequential(
            ConvBlock(filters[0]*4, filters[0]),
            nn.Conv2d(filters[0], num_classes, kernel_size=1) # 클래스 개수만큼 출력
        )

    def forward(self, x):
        # 인코더 및 스킵 연결 경로
        x0_0 = self.conv0_0(x)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(torch.cat([x0_0, self.up(x1_0)], 1))

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(torch.cat([x1_0, self.up(x2_0)], 1))
        x0_2 = self.conv0_2(torch.cat([x0_0, x0_1, self.up(x1_1)], 1))

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(torch.cat([x2_0, self.up(x3_0)], 1))
        x1_2 = self.conv1_2(torch.cat([x1_0, x1_1, self.up(x2_1)], 1))
        x0_3 = self.conv0_3(torch.cat([x0_0, x0_1, x0_2, self.up(x1_2)], 1))

        # 최종 디코더 입력을 위해 모든 x0_i 특징 맵을 결합합니다.
        # 채널 수: 64 + 64 + 64 + 64 = 256
        final_features = torch.cat([x0_0, x0_1, x0_2, x0_3], 1)

        # 최종 출력 계산
        final_denoised = self.decoder1(final_features)
        final_unwrap = self.decoder2(final_features)

        # Deep Supervision이 활성화된 경우, 중간 출력들을 함께 반환
        if self.deep_supervision:
            # 각 중간 단계의 출력을 계산 (예시: 여기서는 생략, 필요시 추가 구현)
            # output1 = self.final1(x0_1)
            # output2 = self.final2(x0_2)
            # output3 = self.final3(x0_3)
            # return [final_denoised, output3, output2, output1], [final_unwrap, ...]
            # 현재는 최종 출력만 반환하는 구조이므로, deep_supervision을 위한
            # 추가적인 출력 레이어가 필요합니다. 아래는 최종 출력만 반환하는 현재 구조입니다.
            pass # 향후 확장을 위해 남겨둠

        return final_denoised, final_unwrap