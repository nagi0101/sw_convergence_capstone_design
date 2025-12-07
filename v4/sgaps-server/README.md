# SGAPS-MAE 서버

## 소개

이 프로젝트는 SGAPS-MAE 모델을 사용하여 희소 픽셀(sparse pixel)로부터 이미지를 생성하는 FastAPI 기반의 백엔드 서버입니다.

## 설치 방법 (Conda 기준)

### 1. Conda 가상 환경 생성 및 활성화

프로젝트의 의존성을 관리하기 위해 Conda 가상 환경을 사용합니다. `sgaps-server`라는 이름의 Python 3.10 환경을 생성합니다.

```bash
conda create --name sgaps-server python=3.10
conda activate sgaps-server
```

### 2. PyTorch 설치

Conda를 사용하면 CUDA 의존성을 포함하여 PyTorch를 더 쉽게 설치할 수 있습니다. **먼저 이 단계를 진행한 후, 다음 단계로 넘어가세요.**

시스템에 맞는 올바른 PyTorch 버전을 설치하는 것이 매우 중요합니다.
자세한 내용은 [PyTorch 공식 홈페이지](https://pytorch.org/get-started/locally/)에서 Conda용 설치 명령을 확인하세요.

**GPU (CUDA) 사용 시:**
NVIDIA GPU가 설치되어 있는 경우, 다음 예시와 같이 PyTorch와 CUDA 툴킷을 함께 설치합니다. (아래는 CUDA 12.1 기준)

```bash
conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
```

**CPU만 사용 시:**
```bash
conda install pytorch torchvision torchaudio cpuonly -c pytorch
```

### 3. 나머지 의존성 설치

PyTorch 설치가 완료되면, `pip`을 사용하여 `requirements.txt` 파일에 명시된 나머지 패키지들을 설치합니다.

```bash
pip install -r requirements.txt
```

## 서버 실행

모든 의존성 설치가 완료되면, uvicorn을 사용하여 서버를 실행할 수 있습니다.

```bash
uvicorn main:app --reload
```

서버가 시작되면, 기본적으로 `http://127.0.0.1:8000/docs` 주소에서 API 문서를 확인할 수 있습니다.
