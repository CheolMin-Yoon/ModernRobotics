# ModernRobotics

Modern Robotics 교재 기반 실습 코드 저장소입니다.

## Git 업로드 방법

### 1. 변경사항 전체 스테이징 (삭제 포함)

```bash
git add -A
```

### 2. 커밋

```bash
git commit -m "커밋 메시지"
```

예시: `git commit -m "upload ch03"`

### 3. 푸시

```bash
git push origin main
```

리모트에 충돌이 있을 경우 (rejected 에러 시):

```bash
git push origin main --force
```

> force push는 리모트를 로컬 상태로 덮어쓰므로 주의해서 사용할 것

---

## 환경 설정

### 1. Conda 가상환경 생성

```bash
conda create -n mr python=3.13
conda activate mr
```

### 2. 패키지 설치

```bash
pip install numpy scipy pinocchio mujoco
```

### 3. 외부 데이터 클론

```bash
# UR5 등 URDF 파일 (비교 검증용)
git clone https://github.com/Daniella1/urdf_files_dataset.git

# MuJoCo 로봇 모델 모음
git clone https://github.com/google-deepmind/mujoco_menagerie.git
```

> 두 폴더 모두 프로젝트 루트에 위치시킵니다. `urdf_files_dataset/`은 `.gitignore`에 등록되어 있습니다.

---

## 폴더 구조

```
ch02_configuration_space/
ch03_rigid_body_motion/
ch04_forward_kinematics/
mujoco_menagerie/        # git clone
urdf_files_dataset/      # git clone (.gitignore)
pin_utils/
```
