import cv2
import numpy as np
import os
from glob import glob



def concat(np_img, np_mask, np_result):
    overlay = cv2.addWeighted(np_img, 0.5, np_mask, 0.5, 0)
    concat = np.concatenate([overlay, np_result], axis=1)
    return concat.astype(np.uint8)


if __name__ == "__main__":
    image_paths = sorted(glob("img/*.png"))
    mask_paths = sorted(glob("mask/*.png"))
    result_paths = sorted(glob("result/*.png"))

    for im, ma, re in zip(image_paths, mask_paths, result_paths):
        _, filename = os.path.split(im)
        im = cv2.imread(im)
        ma = cv2.imread(ma)
        re = cv2.imread(re)

        con = concat(im, ma, re)
        print(filename)
        cv2.imwrite(f"concat/{filename}", con)
        # break
    
