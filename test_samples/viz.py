import cv2
import numpy as np
import os
from glob import glob


dst_dirname = "concat"


def concat(np_img, np_mask, np_result1, np_result2):
    overlay = cv2.addWeighted(np_img, 0.5, np_mask, 0.5, 0)
    concat = np.concatenate([overlay, np_result1, np_result2], axis=1)
    return concat.astype(np.uint8)


if __name__ == "__main__":
    image_paths = sorted(glob("img/*.png"))
    mask_paths = sorted(glob("mask/*.png"))
    result_paths1 = sorted(glob("result_d****-v1_org/*.png"))
    result_paths2 = sorted(glob("result_d****-v1_ft1/*.png"))

    for im, ma, re1, re2 in zip(image_paths, mask_paths, result_paths1, result_paths2):
        _, filename = os.path.split(im)
        im = cv2.imread(im)
        ma = cv2.imread(ma)
        re1 = cv2.imread(re1)
        re2 = cv2.imread(re2)

        con = concat(im, ma, re1, re2)
        print(filename)

        if os.path.exists(dst_dirname) == False:
            os.makedirs(dst_dirname, exist_ok=True)
        cv2.imwrite(f"{dst_dirname}/{filename}", con)
        # break
    
