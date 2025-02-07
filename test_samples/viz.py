import cv2
import numpy as np
import os
from glob import glob


dst_dirname = "concat_backboneA_backboneB_brushnetname"


def concat(np_img, np_mask, np_result0, np_result1):
    overlay = cv2.addWeighted(np_img, 0.5, np_mask, 0.5, 0)
    concat = np.concatenate([overlay, np_result0, np_result1], axis=1)
    return concat.astype(np.uint8)


if __name__ == "__main__":
    image_paths = sorted(glob("img/*.png"))
    mask_paths = sorted(glob("mask/*.png"))

    result_paths0 = sorted(glob("result_backbonename_brushnetname/*.png"))
    result_paths1 = sorted(glob("result_backbonename_brushnetname/*.png"))
    # result_paths2 = sorted(glob("result_backbonename_brushnetname/*.png"))


    for im, ma, re0, re1 in zip(image_paths, mask_paths, result_paths0, result_paths1):
        _, filename = os.path.split(im)
        im = cv2.imread(im)
        ma = cv2.imread(ma)
        
        re0 = cv2.imread(re0)
        re1 = cv2.imread(re1)
        # re2 = cv2.imread(re2)

        con = concat(im, ma, re0, re1)
        print(filename)

        if os.path.exists(dst_dirname) == False:
            os.makedirs(dst_dirname, exist_ok=True)
        cv2.imwrite(f"{dst_dirname}/{filename}", con)
        # break
    
