import cv2
import argparse
import os
from stitch import *
os.chdir("code")
def parse_opt():
    parser = argparse.ArgumentParser()
    parser.add_argument("--undefog-img-path", type=str, help='input img path')
    parser.add_argument("--defog-img-path", type=str, help='input img path')
    parser.add_argument("--fusion-result-name", type=str, help='Save name of the fusion result')
    parser.add_argument("--matching-result-name", type=str, default='matching.jpg',help='Save name of matching result,which is generated by matching algorithm')
    opt = parser.parse_args()
    return opt

def main():
    opt = parse_opt()
    img1 = cv2.imread(opt.undefog_img_path)
    img1=img1[:,0:img1.shape[1]//4*3,:]
    img2 = cv2.imread(opt.defog_img_path)
    img2=img2[:,img2.shape[1]//4:img2.shape[1],:]
    stitch=Image_Stitching().blending(img1,img2,opt.matching_result_name)
    cv2.imwrite(opt.fusion_result_name, stitch)

if __name__ == '__main__':
    main()