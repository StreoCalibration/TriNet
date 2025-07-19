import os
import cv2
import torch
from torch.utils.data import Dataset
import numpy as np
from torchvision import transforms

class DHIDataset(Dataset):
    """
    DHI(디지털 홀로그래피 간섭계) 데이터셋을 위한 커스텀 클래스.
    root_dir 내부에 'noisy', 'gt_denoised', 'gt_unwrap' 폴더가 있다고 가정합니다.
    """
    def __init__(self, root_dir, is_train=True):
        """
        Args:
            root_dir (string): 'noisy', 'gt_denoised', 'gt_unwrap' 폴더를 포함하는 디렉토리 경로.
            is_train (bool): True이면 데이터 증강(augmentation)을 적용합니다.
        """
        self.root_dir = root_dir
        self.is_train = is_train
        self.noisy_dir = os.path.join(root_dir, 'noisy')
        self.gt_denoised_dir = os.path.join(root_dir, 'gt_denoised')
        self.gt_unwrap_dir = os.path.join(root_dir, 'gt_unwrap')

        # 'noisy' 폴더를 기준으로 파일 목록을 생성합니다.
        self.image_files = sorted([f for f in os.listdir(self.noisy_dir) if f.endswith(('.png', '.jpg', '.tif', '.tiff'))])

        # 입력 이미지를 위한 변환: 텐서로 바꾸고 [-1, 1] 범위로 정규화
        self.input_transform = transforms.Compose([
            transforms.ToTensor(), # 값의 범위를 [0, 1]로 스케일링 및 CHW 형식으로 변경
            transforms.Normalize(mean=[0.5], std=[0.5]) # [0, 1] -> [-1, 1]
        ])

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_name = self.image_files[idx]

        # 3가지 이미지의 전체 경로를 구성합니다.
        noisy_path = os.path.join(self.noisy_dir, img_name)
        gt_denoised_path = os.path.join(self.gt_denoised_dir, img_name)
        gt_unwrap_path = os.path.join(self.gt_unwrap_dir, img_name)

        # PR-01 요구사항: 8-bit 또는 16-bit 입력을 처리하기 위해 IMREAD_UNCHANGED 사용
        noisy_img = cv2.imread(noisy_path, cv2.IMREAD_UNCHANGED)
        gt_denoised = cv2.imread(gt_denoised_path, cv2.IMREAD_UNCHANGED)
        gt_unwrap_map = cv2.imread(gt_unwrap_path, cv2.IMREAD_UNCHANGED)

        # 이미지가 정상적으로 로드되었는지 확인
        if noisy_img is None or gt_denoised is None or gt_unwrap_map is None:
            raise FileNotFoundError(f"이미지 로딩 실패: {img_name}")

        # 입력 및 노이즈 제거 정답 이미지를 텐서로 변환 및 정규화
        # 모델의 Tanh 활성화 함수 출력 범위인 [-1, 1]에 맞추기 위함입니다.
        noisy_img_tensor = self.input_transform(noisy_img)
        gt_denoised_tensor = self.input_transform(gt_denoised)

        # 언래핑 정답 맵은 클래스 레이블이므로 LongTensor로 변환 (정규화 X)
        gt_unwrap_map_tensor = torch.from_numpy(gt_unwrap_map.astype(np.int64))

        # 데이터 증강 (Data Augmentation): 훈련 데이터에만 적용
        # 입력과 정답 데이터에 동일한 변환을 적용해야 합니다.
        if self.is_train:
            # 예시: 50% 확률로 좌우 반전
            if torch.rand(1) < 0.5:
                noisy_img_tensor = transforms.functional.hflip(noisy_img_tensor)
                gt_denoised_tensor = transforms.functional.hflip(gt_denoised_tensor)
                gt_unwrap_map_tensor = transforms.functional.hflip(gt_unwrap_map_tensor)

            # 예시: 50% 확률로 상하 반전
            if torch.rand(1) < 0.5:
                noisy_img_tensor = transforms.functional.vflip(noisy_img_tensor)
                gt_denoised_tensor = transforms.functional.vflip(gt_denoised_tensor)
                gt_unwrap_map_tensor = transforms.functional.vflip(gt_unwrap_map_tensor)

        return noisy_img_tensor, gt_denoised_tensor, gt_unwrap_map_tensor