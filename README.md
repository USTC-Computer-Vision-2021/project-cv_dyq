# 基于 图像配准 的 Photoshop前后的图像融合

成员及分工
- 张三 PBxxxxx
  - 设计，调研，coding


## 问题描述

- 该实验方向的选择动机来源于个人的摄影爱好，摄影中常需要对拍摄的图像进行后期处理，以得到色彩表达更强烈、饱和度更好、观感质量更好的图像。在平时拍摄过程中，由于天气状况常常会得到一些“受污染”的图像，比如在雾霾天气下拍摄到了取景满意的图像，在后期中常采用photoshop的去霾功能进行质量改善。我考虑是否可以应用计算机视觉的技术，将photoshop处理前的雾霾图像，和photoshop去霾后的图像融合到一起，来获得一种新的摄影艺术效果。结合本课程学习的内容，该创意在算法方面可以体现为基于图像配准的图像融合。

## 原理分析

### 1、图像配准的基本过程：
大量的特征将在第一张源图中被提取出来，这些特征将在目标图像中寻找匹配的特征信息。通过两幅图片中相匹配的特征信息，源图和目标图像之间的像素坐标转换关系将会被提取出来。借助这种转换关系可以实现将一幅图片与另一幅校准对齐，这种转换关系可以用单应矩阵来表示。
### 2、单应矩阵：
在计算机视觉中，平面的单应性被定义为一个平面到另外一个平面的投影映射。单应性简单来说就是一个3*3 矩阵：

![image]( https://github.com/USTC-Computer-Vision-2021/project-cv_dyq/blob/main/funcImg/func1.JPG)

若（x1,y1）是第一幅图中的一个点的坐标，（x2,y2）是第二幅图片中相同物理点的坐标，单应矩阵可以通过以下方式将这两幅图片联系起来：

![image]( https://github.com/USTC-Computer-Vision-2021/project-cv_dyq/blob/main/funcImg/func2.JPG)

如果能获取这个单应矩阵，那么应用这个单应矩阵对一幅图片所有像素的坐标进行变换，变换结果就能和第二图图片配准。
### 3、单应矩阵：
可以使用opencv函数findHomography计算单应矩阵，将存储对应点的数据做函数输入，输出就是单应矩阵。那么该如何找到对应点。
### 4、寻找匹配点：
在计算机视觉应用中，我们通常需要识别图像中感兴趣的稳定点。这些点被称为关键点或特征点。一个特征点检测器包括两方面：
- 定位器：图像中被检测到的点在图像尺度变化，图像旋转等条件下要保持稳定。定位器就是用来找到这些稳定的点。
- 描述子：定位器告诉我们兴趣点的位置，描述子则可以使我们区分不同的兴趣点。描述器可以把每个特征点描述成一个由数字构成的矩阵，理想情况下，在两幅图像中，相同物理点的描述子结果是相同的。
匹配算法用于寻找在两幅图片中的那些对应特征点。为此，将一个图像中每个特征的描述符与第二个图像中每个特征的描述符进行比较，以找到良好的匹配。本次使用sift作为特征检测器。

## 代码实现

单应矩阵计算代码：
```python
def registration(self,img1,img2,matchResName):
    kp1, des1 = self.sift.detectAndCompute(img1, None)
    kp2, des2 = self.sift.detectAndCompute(img2, None)
    matcher = cv2.BFMatcher()
    raw_matches = matcher.knnMatch(des1, des2, k=2)
    good_points = []
    good_matches=[]
    for m1, m2 in raw_matches:
        if m1.distance < self.ratio * m2.distance:
            good_points.append((m1.trainIdx, m1.queryIdx))
            good_matches.append([m1])
    good_matches=good_matches[::16]
    img3 = cv2.drawMatchesKnn(img1, kp1, img2, kp2, good_matches, None, flags=2)
    cv2.imwrite(matchResName, img3)
    if len(good_points) > self.min_match:
        image1_kp = np.float32(
            [kp1[i].pt for (_, i) in good_points])
        image2_kp = np.float32(
            [kp2[i].pt for (i, _) in good_points])
        H, status = cv2.findHomography(image2_kp, image1_kp, cv2.RANSAC,5.0)
    return H
```
图像融合代码：
```python
def blending(self,img1,img2,matchResName):
    H = self.registration(img1,img2,matchResName)
    height_img1 = img1.shape[0]
    width_img1 = img1.shape[1]
    width_img2 = img2.shape[1]
    height_panorama = height_img1
    width_panorama = width_img1 +width_img2

    panorama1 = np.zeros((height_panorama, width_panorama, 3))
    mask1 = self.create_mask(img1,img2,version='left_image')
    panorama1[0:img1.shape[0], 0:img1.shape[1], :] = img1
    panorama1 *= mask1
    mask2 = self.create_mask(img1,img2,version='right_image')
    panorama2 = cv2.warpPerspective(img2, H, (width_panorama, height_panorama))*mask2
    result=panorama1+panorama2

    rows, cols = np.where(result[:, :, 0] != 0)
    min_row, max_row = min(rows), max(rows) + 1
    min_col, max_col = min(cols), max(cols) + 1
    final_result = result[min_row:max_row, min_col:max_col, :]
    return final_result
```

## 效果展示

某融合结果如下,左半边为处理后,右半边为处理前：

![image]( https://github.com/USTC-Computer-Vision-2021/project-cv_dyq/blob/main/ouput/tower_compare.jpg)

## 工程结构

```text
.
├── code
│   ├── run.py
│   └── stitch.py
├── input
│   ├── tower1.jpg
│   └── tower2.jpg
└── output
│   ├── tower_compare.jpg
│   └── tower_matching.jpg
```

## 运行说明

依赖环境和库的具体版本号, requirements.txt内容如下：

```
numpy
opencv-contrib-python==4.5.4.60
opencv-python==4.5.1.48
```

运行说明：
```
python run.py --undefog-img-path ../input/tower1.jpg --defog-img-path ../input/tower2.jpg --fusion-result-name ../output/tower_compare.jpg --matching-result-name ../output/tower_matching.jpg
```

