## 📄 제품 요구사항 명세서 (PRD): DHI 위상 복원 TriNet 모듈

### 1. 개요 (Introduction)

전통적인 디지털 홀로그래피 간섭계(DHI)의 위상 복원 알고리즘은 노이즈에 민감하고, 위상 펼치기(phase unwrapping) 과정에서 발생하는 오류로 인해 정확한 3D 변형 측정에 한계가 있다. 본 문서는 딥러NING 기반의 **TriNet 아키텍처를 활용하여, 노이즈가 포함된 간섭 무늬 이미지로부터 노이즈 제거와 위상 펼치기를 동시에 수행**하는 고성능 위상 복원 모듈을 개발하는 것을 목표로 한다.

### 2. 목표 (Objectives)

- **정확도 향상**: 기존 필터링 및 위상 펼치기 알고리즘 대비 **PSNR(Peak Signal-to-Noise Ratio) 15% 이상 개선**.
    
- **프로세스 통합**: 노이즈 제거와 위상 펼치기 프로세스를 단일 네트워크에서 처리하여 파이프라인을 단순화하고 효율성 증대.
    
- **처리 속도 확보**: 1024x1024 해상도 이미지 기준, GPU(NVIDIA RTX 3080급)에서 **추론 시간 50ms 이내** 달성.
    

### 3. 사용자 (Target Users)

- **연구 개발 엔지니어**: DHI, DHM(디지털 홀로그래피 현미경) 장비를 이용해 정밀 측정을 수행하는 연구원.
    
- **AOI 장비 개발자**: 반도체 웨이퍼, PCB 등 정밀 부품의 결함 검사 장비에 탑재할 3D 형상 복원 알고리즘 개발자.
    

### 4. 기능 요구사항 (Functional Requirements)

|ID|기능|상세 설명|
|---|---|---|
|**FR-01**|**입력 데이터 처리**|8-bit 또는 16-bit Grayscale의 간섭 무늬 이미지(Raw Interferogram)를 입력받는다.|
|**FR-02**|**동시 복원 수행**|단일 추론(inference)을 통해 **(1) 노이즈가 제거된 위상 맵**과 **(2) 펼쳐진 절대 위상 맵**을 동시에 출력한다.|
|**FR-03**|**출력 데이터 형식**|출력되는 위상 맵은 32-bit Float 형식의 배열 또는 이미지 파일(e.g., TIFF, PFM)로 제공되어야 한다.|
|**FR-04**|**모델 관리**|사전 학습된 TriNet 모델 가중치 파일(`.pth`, `.onnx` 등)을 로드하여 사용할 수 있는 인터페이스를 제공한다.|

### 5. 비기능 요구사항 (Non-functional Requirements)

|ID|구분|상세 설명|
|---|---|---|
|**NFR-01**|**성능**|목표 섹션의 처리 속도 및 정확도 기준을 만족해야 한다.|
|**NFR-02**|**플랫폼 호환성**|Windows 10/11 및 Linux (Ubuntu 20.04 이상) 환경에서 동작해야 한다.|
|**NFR-03**|**구현 기술**|코어 로직은 **C++**로 개발하며, 이미지 처리는 **OpenCV**, 딥러닝 추론은 **LibTorch** 또는 **ONNX Runtime**을 사용한다.|
|**NFR-04**|**메모리 사용량**|추론 시 GPU VRAM 사용량은 8GB 이하여야 한다.|

---

## 