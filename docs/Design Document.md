🛠️ 기술 설계 문서 (Design Document): TriNet 모델 구현

### 1. 시스템 아키텍처 (System Architecture)

```
+------------------+     +-----------------------+     +--------------------------+
|                  | --> |                       | --> |                          |
|  Image Input     |     |  TriNet Inference     |     |  Dual Phase Map Output   |
|  (C++/OpenCV)    |     |  (LibTorch/ONNX)      |     |  (Denoised & Unwrapped)  |
|                  |     |                       |     |                          |
+------------------+     +----------+------------+     +--------------------------+
                                     |
                             +-------+-------+
                             |               |
                             |  Pre-trained  |
                             |  Model Weights|
                             |               |
                             +---------------+
```

- **Image Input**: OpenCV를 사용하여 간섭 무늬 이미지를 로드하고 정규화 등 전처리 수행.
    
- **TriNet Inference**: 전처리된 텐서를 입력받아 LibTorch/ONNX Runtime을 통해 TriNet 모델 추론 실행.
    
- **Dual Phase Map Output**: 모델로부터 출력된 2개의 텐서를 후처리하여 최종적인 위상 맵 이미지로 변환 및 저장.
    

### 2. TriNet 모델 상세 설계 (Vengala et al. 기반)

- **공통 인코더 (Shared Encoder)**:
    
    - Pyramidal 구조로, 연속적인 Convolution Layer와 Down-sampling(Max-pooling)으로 구성.
        
    - 각 Conv Block: `Conv2d(kernel=3, padding=1)` + `BatchNorm2d` + `ReLU`.
        
    - 입력 이미지로부터 계층적인 특징(hierarchical features)을 추출. 각 스케일의 특징 맵은 디코더의 스킵 커넥션에 사용됨.
        
- **이중 디코더 (Dual Decoders)**:
    
    - 인코더의 최종 특징 맵을 입력받아 두 개의 독립적인 경로로 분기.
        
        1. **Decoder 1 (Denoising)**: 노이즈 제거된 (wrapped) 위상 맵 복원.
            
        2. **Decoder 2 (Unwrapping)**: 펼쳐진 (unwrapped) 절대 위상 맵 복원.
            
    - 각 디코더는 Up-sampling(`ConvTranspose2d`)과 스킵 커넥션(Concatenation)을 통해 공간 해상도를 복원.
        
    - 최종 출력 레이어는 `1x1 Conv2d`를 사용하여 채널 수를 1로 맞춤.
        
- **다중 작업 손실 함수 (Multi-task Loss Function)**:
    
    - 두 디코더의 출력을 동시에 최적화하기 위해 각 작업의 손실을 가중 합산하여 사용.
        
    - Ltotal​=w1​⋅Ldenoise​+w2​⋅Lunwrap​
        
    - 여기서 $L_{denoise}$와 $L_{unwrap}$은 각각 Ground Truth와의 **MSE (Mean Squared Error)** 또는 **L1 Loss**를 사용할 수 있으며, 가중치 w_1, w_2는 실험을 통해 최적값을 결정.
        

### 3. 데이터 파이프라인 (Data Pipeline)

- **학습 데이터**: 시뮬레이션을 통해 생성된 다양한 노이즈 레벨과 변형 형태를 가진 간섭 무늬 이미지 및 해당 Ground Truth(깨끗한 위상 맵, 펼쳐진 위상 맵) 쌍.
    
- **전처리**:
    
    - 이미지 크기를 모델 입력에 맞게 리사이즈 (e.g., 512x512).
        
    - 픽셀 값을 `[0, 1]` 또는 `[-1, 1]` 범위로 정규화.
        
    - Data Augmentation: Random Flip, Rotation 등을 적용하여 모델의 강건성 확보.
        

### 4. 구현 계획 (Implementation Plan)

- **주요 클래스 (C++)**:
    
    - `TriNetInference`: 모델 로딩, 전/후처리 및 추론 실행을 담당하는 메인 클래스.
        
    - `ImagePreprocessor`: OpenCV를 사용한 이미지 정규화 및 텐서 변환 클래스.
        
    - `ModelConfig`: 모델 경로, 입력/출력 노드 이름 등 설정을 관리하는 구조체.
        
- **개발 순서**:
    
    1. Python(PyTorch)으로 모델 아키텍처 프로토타이핑 및 학습 수행.
        
    2. 학습된 모델을 TorchScript 또는 ONNX 형식으로 변환.
        
    3. C++ 환경에서 LibTorch/ONNX Runtime을 이용해 모델 로딩 및 추론 기능 구현.
        
    4. OpenCV를 연동하여 전체 파이프라인 완성.
        

### 5. 검증 및 테스트 (Validation & Testing)

- **단위 테스트**: 각 클래스(전처리기, 추론기)의 기능이 독립적으로 정확히 동작하는지 검증.
    
- **통합 테스트**: 전체 파이프라인이 실제 간섭 무늬 이미지에 대해 예상된 출력(2개의 위상 맵)을 생성하는지 확인.
    
- **성능 평가**: 시뮬레이션 및 실제 데이터로 구성된 별도의 테스트셋을 이용하여 아래 지표를 측정.
    
    - **정확도**: PSNR, SSIM, MSE.
        
    - **처리 속도**: 이미지당 평균 추론 시간(ms).