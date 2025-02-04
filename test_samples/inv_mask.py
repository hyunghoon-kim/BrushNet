from glob import glob
import os
import cv2


output_dir = "./inv_mask"
filepaths = glob("./mask/*")

if os.path.exists(output_dir) == False:
    os.makedirs(output_dir, exist_ok=True)

for filepath in filepaths:
    # 마스크 이미지를 그레이스케일로 읽기 (바이너리 마스크이므로 단일 채널)
    mask = cv2.imread(filepath, cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Error reading {filepath}")
        continue

    # 마스크 인버트: 각 픽셀 값을 255에서 빼기 (0<->255, 255<->0)
    inv_mask = 255 - mask

    # 저장할 파일명 생성
    filename = os.path.basename(filepath)
    output_path = os.path.join(output_dir, filename)

    # 인버트된 마스크 저장
    cv2.imwrite(output_path, inv_mask)
    print(f"Inverted mask saved to {output_path}")

