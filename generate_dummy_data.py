import os
import numpy as np
import cv2
from tqdm import tqdm

def create_dummy_phase_data(output_dir, num_samples, img_size=512, num_classes=11):
    """
    가상의 위상 맵 데이터셋을 생성합니다.

    Args:
        output_dir (str): 데이터셋을 저장할 루트 디렉토리.
        num_samples (int): 생성할 샘플의 수.
        img_size (int): 이미지의 크기 (height, width).
        num_classes (int): 랩 카운트 맵의 클래스 수 (e.g., 11 -> -5 to +5).
    """
    # 데이터셋 폴더 구조 생성
    noisy_dir = os.path.join(output_dir, 'noisy')
    gt_denoised_dir = os.path.join(output_dir, 'gt_denoised')
    gt_unwrap_dir = os.path.join(output_dir, 'gt_unwrap')
    os.makedirs(noisy_dir, exist_ok=True)
    os.makedirs(gt_denoised_dir, exist_ok=True)
    os.makedirs(gt_unwrap_dir, exist_ok=True)

    print(f"'{output_dir}'에 {num_samples}개의 가상 데이터를 생성합니다...")

    wrap_count_offset = num_classes // 2 # 11 -> 5

    for i in tqdm(range(num_samples), desc=f"Generating data in {output_dir}"):
        # 1. 절대 위상 맵 (Ground Truth Absolute Phase) 생성
        #    - 간단한 예시로 X, Y 좌표를 이용한 원뿔(cone) 형태를 생성합니다.
        x = np.linspace(-1, 1, img_size)
        y = np.linspace(-1, 1, img_size)
        xx, yy = np.meshgrid(x, y)
        
        # 랜덤한 패턴을 위해 파라미터 추가
        rand_freq_x = np.random.uniform(2, 8)
        rand_freq_y = np.random.uniform(2, 8)
        rand_phase = np.random.uniform(0, np.pi)
        
        absolute_phase = np.sqrt((rand_freq_x * xx)**2 + (rand_freq_y * yy)**2) * np.pi + rand_phase
        
        # 2. 노이즈 없는 wrapped 위상 맵 (gt_denoised) 생성: [-pi, pi]
        gt_denoised_rad = np.angle(np.exp(1j * absolute_phase))
        
        # 3. 랩 카운트 맵 (gt_unwrap) 생성: k = (psi - phi) / 2pi
        k = np.round((absolute_phase - gt_denoised_rad) / (2 * np.pi))
        gt_unwrap_map = np.clip((k + wrap_count_offset), 0, num_classes - 1).astype(np.int64)

        # 4. 노이즈 포함된 wrapped 위상 맵 (noisy) 생성
        noise = np.random.normal(0, 0.4, gt_denoised_rad.shape)
        noisy_rad = np.angle(np.exp(1j * (gt_denoised_rad + noise)))

        # 5. 이미지를 8-bit Grayscale로 변환하여 저장: [-pi, pi] -> [0, 255]
        def to_uint8(rad_img):
            return ((rad_img + np.pi) / (2 * np.pi) * 255).astype(np.uint8)

        noisy_img_8bit = to_uint8(noisy_rad)
        gt_denoised_8bit = to_uint8(gt_denoised_rad)
        gt_unwrap_map_8bit = gt_unwrap_map.astype(np.uint8)

        filename = f"{i:04d}.png"
        cv2.imwrite(os.path.join(noisy_dir, filename), noisy_img_8bit)
        cv2.imwrite(os.path.join(gt_denoised_dir, filename), gt_denoised_8bit)
        cv2.imwrite(os.path.join(gt_unwrap_dir, filename), gt_unwrap_map_8bit)

if __name__ == '__main__':
    # 훈련용 및 검증용 가상 데이터 생성
    create_dummy_phase_data('datasets/dummy_train', num_samples=200, img_size=512)
    create_dummy_phase_data('datasets/dummy_validation', num_samples=50, img_size=512)
    print("\n가상 데이터셋 생성이 완료되었습니다.")