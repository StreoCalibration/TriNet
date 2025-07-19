import torch
import cv2
import numpy as np
from torchvision import transforms
# from model import TriNet

# --- 1. 모델 및 데이터 준비 ---
device = torch.device('cpu') # 추론은 CPU에서도 가능
model = TriNet(in_channels=1, num_classes=11)
model.load_state_dict(torch.load('trinet_model.pth', map_location=device))
model.eval()

# 이미지 전처리
transform = transforms.Compose([
    transforms.ToTensor(), # [0, 255] -> [0, 1] & HWC -> CHW
    transforms.Normalize(mean=[0.5], std=[0.5]) # [0, 1] -> [-1, 1]
])
image = cv2.imread('path/to/your/interferogram.png', cv2.IMREAD_GRAYSCALE)
input_tensor = transform(image).unsqueeze(0) # 배치 차원 추가

# --- 2. 추론 실행 ---
with torch.no_grad():
    pred_denoised, pred_unwrap = model(input_tensor)

# --- 3. 후처리 및 최종 위상 결합 ---
# 이 단계는 딥러닝 결과와 물리식을 결합하는 핵심 [cite: 112, 113]
# Denoised 결과 처리 ([-1, 1] -> [0, 255])
phi_denoised_tensor = pred_denoised.squeeze(0).cpu()
phi_denoised_norm = (phi_denoised_tensor * 0.5) + 0.5 # [-1, 1] -> [0, 1]
phi_denoised = (phi_denoised_norm.numpy().transpose(1, 2, 0) * 255).astype(np.uint8)

# Unwrap 결과 처리 (랩 카운트 맵 생성)
k_pred_tensor = torch.argmax(pred_unwrap.squeeze(0), dim=0).cpu().numpy()
# 클래스 인덱스를 실제 랩 카운트 값으로 변환 (예: 0~10 -> -5~5)
wrap_count_offset = 5
k_pred = k_pred - wrap_count_offset

# 최종 절대 위상 계산
# ψ_final = φ_denoised + 2π * k_pred
# 위 계산을 위해 φ_denoised를 float 및 라디안 단위로 변환해야 함
phi_denoised_rad = (phi_denoised_norm.squeeze().numpy() * 2 * np.pi) - np.pi
psi_final = phi_denoised_rad + (2 * np.pi * k_pred)

# 결과 저장 또는 시각화
cv2.imwrite('denoised_phase.png', phi_denoised)
# psi_final은 matplotlib 등으로 시각화