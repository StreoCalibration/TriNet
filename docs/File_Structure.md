프로젝트 구조
trinet/
├── docs/				   # 프로젝트 관련 문서
├── data/                  # 데이터셋 저장 폴더
│   ├── train/
│   └── test/
│
├── train.py               # 모델 학습 스크립트
├── inference.py           # 단일 이미지 추론 스크립트
├── model.py               # TriNet 아키텍처 정의
├── dataset.py             # 사용자 정의 데이터셋 클래스
└── utils.py               # 전/후처리 등 유틸리티 함수