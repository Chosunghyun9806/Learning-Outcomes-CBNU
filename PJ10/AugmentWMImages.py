# 데이터 증강 처리
import os
import shutil
import random
import cv2
import numpy as np
from datetime import datetime


# 로그 함수 정의
def log(message):
    message = f'[{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}] {message}'
    print(message)


# 데이터 증강 함수 정의
def augment_image(image, ratio):
    rows, cols, _ = image.shape  # 이미지 형태를 가져옴

    # 무작위 회전
    if random.random() < ratio['rotation']:
        angle = random.randint(-10, 10)
        m = cv2.getRotationMatrix2D((cols / 2, rows / 2), angle, 1)
        image = cv2.warpAffine(image, m, (cols, rows))

    # 좌우 대칭 및 너비 이동
    if random.random() < ratio['flip_and_shift']:
        if random.random() > 0.5:
            image = cv2.flip(image, 1)

    # 높이 이동
    if random.random() < ratio['height_shift']:
        height_shift = random.randint(-int(rows * 0.15), int(rows * 0.15))
        m = np.float32([[1, 0, 0], [0, 1, height_shift]])
        image = cv2.warpAffine(image, m, (cols, rows))

    # 전단 범위
    if random.random() < ratio['shear']:
        shear_range = random.randint(-10, 10)
        pts1 = np.float32([[5, 5], [20, 5], [5, 20]])
        pt1 = 5 + shear_range * np.random.uniform() - shear_range / 2
        pt2 = 20 + shear_range * np.random.uniform() - shear_range / 2
        pts2 = np.float32([[pt1, 5], [pt2, pt1], [5, pt2]])
        m = cv2.getAffineTransform(pts1, pts2)
        image = cv2.warpAffine(image, m, (cols, rows))

    # 채널 이동 및 확대/축소
    if random.random() < ratio['channel_shift']:
        channel_shift_val = random.randint(-20, 20)
        image = image + channel_shift_val
        image = np.clip(image, 0, 255)

    return image


# 데이터 증강 함수 (10,000개 제한)
def augment_data(images, target, ratio):
    augment_count = len(images)
    while augment_count < target:
        image = random.choice(images)
        augmented_image = augment_image(image, ratio)
        images.append(augmented_image)
        augment_count += 1
        # 증강된 이미지 수가 목표치 도달 하면 종료
        if augment_count >= target:
            break
    return images, augment_count


# 폴더 내 이미지 불러와 리스트 반환 함수 정의
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        img = cv2.imread(img_path)
        if img is not None:
            images.append(img)
    return images


# 데이터 셋 증강 폴더로 복사
def copy_origin_images(image_folder, target_folder, dataset_folders, label_list):

    # 먼저 대상 디렉토리가 존재하는지 확인하고 있다면 삭제
    if os.path.exists(target_folder):
        shutil.rmtree(target_folder)

    for dataset_folder in dataset_folders:
        log(f"Processing Copy Folder : {dataset_folder}")
        for label in label_list:
            # 소스 디렉토리와 대상 디렉토리 경로 설정
            source_label_directory = os.path.join(image_folder, dataset_folder, label)
            target_label_directory = os.path.join(target_folder, label)

            # 대상 레이블 디렉토리가 없는 경우 생성
            if not os.path.exists(target_label_directory):
                os.makedirs(target_label_directory)

            # 소스 레이블 디렉토리의 파일들을 대상 레이블 디렉토리로 복사
            for filename in os.listdir(source_label_directory):
                source_file = os.path.join(source_label_directory, filename)
                target_file = os.path.join(target_label_directory, filename)
                shutil.copyfile(source_file, target_file)


# 원본 데이터 폴더 경로 설정
original_data_dir = 'wm_images'

# 원본 데이터를 레이블에 맞게 복사할 경로 설정
copy_data_dir = 'copy_images'

# 증강 폴더 경로 설정
augmented_data_dir = 'augmented_images'

# 테스트, 학습, 검증 데이터 폴더경로
dataset_dir = ['train']

# 레이블 리스트
labels = os.listdir(os.path.join(original_data_dir, 'train'))

# 각 증강 기법의 비율 설정
augmentation_ratio = {
    'rotation': 0.2,
    'flip_and_shift': 0.2,
    'height_shift': 0.15,
    'shear': 0.1,
    'channel_shift': 0.1
}

# 원본 데이터 증강 폴더로 전체 복사
# 1회성 복사 함수.. 한번 완료 했으면? 이 함수는 주석 처리
copy_origin_images(original_data_dir, copy_data_dir, dataset_dir, labels)


# 먼저 대상 증강 디렉토리가 존재하는지 확인하고 있다면 삭제
if os.path.exists(augmented_data_dir):
    shutil.rmtree(augmented_data_dir)

# 레이블 별 데이터 증강 및 저장
for label in labels:
    log(f"Processing images for label: {label}")

    # 레이블 폴더 경로 설정
    label_folder = os.path.join(copy_data_dir, label)

    # 레이블 폴더 내 이미지 불러옴
    label_images = load_images_from_folder(label_folder)

    # 데이터 개수 확인
    num_images = len(label_images)
    log(f"Number of images: {num_images}")

    # 10,000개 이상인 경우 무작위 데이터 10,000개 선택
    if num_images >= 10000:
        augmented_images = random.sample(label_images, 10000)
        log(f"Selected 10,000 original images for label {label}.")
    else:
        # 10,000개 미만인 경우 원본 데이터를 그대로 복사
        log(f"Copying original {num_images} images for label {label}.")
        selected_images = label_images.copy()
        while len(selected_images) <= 10000:
            selected_images.extend(label_images)

        # 원본 데이터 중 모자란 부분에 대해서만 데이터 증강 수행
        target_count = 10000
        augmented_images, count = augment_data(label_images, target_count, augmentation_ratio)
        log(f"Augmented {count - num_images} images for label {label}.")

    # 증강된 이미지를 저장할 폴더 경로 설정
    augmented_label_folder = os.path.join(augmented_data_dir, label)
    os.makedirs(augmented_label_folder, exist_ok=True)

    # 증강된 이미지를 레이블 폴더에 저장
    for idx, augmented_img in enumerate(augmented_images):
        cv2.imwrite(os.path.join(augmented_label_folder, f'{idx}.bmp'), augmented_img)
