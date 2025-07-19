import torch
import torch.optim as optim
from torch.utils.data import DataLoader
# from model import TriNet # 위에서 정의한 모델
# from dataset import DHIDataset # 사용자 정의 데이터셋

# --- 1. 설정 및 준비 ---
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = TriNet(in_channels=1, num_classes=11).to(device)
# train_dataset = DHIDataset(...)
# train_loader = DataLoader(train_dataset, batch_size=4, shuffle=True)

# --- 2. 손실 함수 및 옵티마이저 ---
# 논문의 멀티태스크 학습 방식에 따라 손실 함수를 정의 [cite: 7]
criterion_denoise = nn.L1Loss()
criterion_unwrap = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=1e-4)
w1, w2 = 0.5, 0.5 # 손실 가중치 (Hyperparameter)

# --- 3. 학습 루프 ---
model.train()
for epoch in range(NUM_EPOCHS):
    for noisy_img, gt_denoised, gt_unwrap_map in train_loader:
        noisy_img = noisy_img.to(device)
        gt_denoised = gt_denoised.to(device)
        gt_unwrap_map = gt_unwrap_map.to(device)

        # 순전파
        pred_denoised, pred_unwrap = model(noisy_img)

        # 손실 계산
        loss_d = criterion_denoise(pred_denoised, gt_denoised)
        loss_u = criterion_unwrap(pred_unwrap, gt_unwrap_map)
        loss = w1 * loss_d + w2 * loss_u

        # 역전파 및 가중치 업데이트
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")

# --- 4. 모델 저장 ---
torch.save(model.state_dict(), 'trinet_model.pth')