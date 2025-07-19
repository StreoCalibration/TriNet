import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
import numpy as np
from model import TriNet
from dataset import DHIDataset # 새로 만든 DHIDataset 클래스를 임포트

# --- 1. 설정 및 준비 ---
NUM_EPOCHS = 50
# BATCH_SIZE를 줄여 메모리 사용량을 낮춥니다.
BATCH_SIZE = 2
# 그래디언트 누적 스텝을 설정합니다. Effective Batch Size = BATCH_SIZE * ACCUMULATION_STEPS
ACCUMULATION_STEPS = 2
LEARNING_RATE = 1e-4

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TriNet(in_channels=1, num_classes=11).to(device)

# 실제 데이터셋 경로를 지정하고 데이터로더를 생성합니다.
# 기본적으로 가상 데이터셋 경로를 사용합니다.
TRAIN_DATA_PATH = "datasets/dummy_train"
VAL_DATA_PATH = "datasets/dummy_validation"

# 데이터셋 존재 여부 확인
if not os.path.exists(TRAIN_DATA_PATH) or not os.path.exists(VAL_DATA_PATH):
    print(f"'{TRAIN_DATA_PATH}' 또는 '{VAL_DATA_PATH}' 경로를 찾을 수 없습니다.")
    print("먼저 'generate_dummy_data.py' 스크립트를 실행하여 가상 데이터셋을 생성해주세요.")
    exit()

train_dataset = DHIDataset(root_dir=TRAIN_DATA_PATH, is_train=True)
val_dataset = DHIDataset(root_dir=VAL_DATA_PATH, is_train=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# --- 2. 손실 함수 및 옵티마이저 ---
# 논문의 멀티태스크 학습 방식에 따라 손실 함수를 정의 [cite: 7]
criterion_denoise = nn.L1Loss()
criterion_unwrap = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
w1, w2 = 0.5, 0.5  # 손실 가중치 (Hyperparameter)

# Automatic Mixed Precision (AMP)를 위한 GradScaler 초기화
# CUDA 사용이 가능할 때만 활성화합니다.
scaler = torch.cuda.amp.GradScaler(enabled=torch.cuda.is_available())

# --- 3. 학습 루프 ---
best_val_loss = np.inf

for epoch in range(NUM_EPOCHS):
    model.train()  # 학습 모드
    epoch_train_loss = 0.0
    optimizer.zero_grad()  # 그래디언트 누적을 위해 루프 시작 전 초기화

    for i, (noisy_img, gt_denoised, gt_unwrap_map) in enumerate(train_loader):
        noisy_img = noisy_img.to(device)
        gt_denoised = gt_denoised.to(device)
        gt_unwrap_map = gt_unwrap_map.to(device)

        # AMP를 위한 autocast 컨텍스트 매니저 사용
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=torch.cuda.is_available()):
            # 순전파
            pred_denoised, pred_unwrap = model(noisy_img)

            # 손실 계산
            loss_d = criterion_denoise(pred_denoised, gt_denoised)
            loss_u = criterion_unwrap(pred_unwrap, gt_unwrap_map)
            loss = w1 * loss_d + w2 * loss_u

            # 그래디언트 누적을 위해 loss를 스텝 수로 나눔
            loss = loss / ACCUMULATION_STEPS

        # 역전파 (scaler가 loss 스케일링을 자동으로 처리)
        scaler.scale(loss).backward()

        # ACCUMULATION_STEPS 마다 가중치 업데이트 및 그래디언트 초기화
        if (i + 1) % ACCUMULATION_STEPS == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        epoch_train_loss += loss.item() * ACCUMULATION_STEPS # 원래 loss 값을 더하기 위해 다시 곱함

    # --- 검증 루프 추가 ---
    model.eval()  # 평가 모드
    epoch_val_loss = 0.0
    with torch.no_grad():
        for noisy_img, gt_denoised, gt_unwrap_map in val_loader:
            noisy_img = noisy_img.to(device)
            gt_denoised = gt_denoised.to(device)
            gt_unwrap_map = gt_unwrap_map.to(device)

            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=torch.cuda.is_available()):
                pred_denoised, pred_unwrap = model(noisy_img)
                loss_d = criterion_denoise(pred_denoised, gt_denoised)
                loss_u = criterion_unwrap(pred_unwrap, gt_unwrap_map)
                loss = w1 * loss_d + w2 * loss_u
            epoch_val_loss += loss.item()

    avg_train_loss = epoch_train_loss / len(train_loader)
    avg_val_loss = epoch_val_loss / len(val_loader)
    print(f"Epoch {epoch+1}/{NUM_EPOCHS}, Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    # --- 4. 최적 모델 저장 ---
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), 'trinet_model_best.pth')
        print(f"Best model saved at epoch {epoch+1} with validation loss: {best_val_loss:.4f}")

print("Training finished.")