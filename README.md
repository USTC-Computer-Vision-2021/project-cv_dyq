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
在计算机视觉中，平面的单应性被定义为一个平面到另外一个平面的投影映射。单应性简单来说就是一个3*3矩阵：
![image]( https://github.com/lexsaints/powershell/blob/master/IMG/ps2.png)

如果需要在编辑器中插入数学公式，可以使用两个美元符 $$ 包裹 LaTeX 格式的数学公式来实现。如输入：
```
$$
\lim_{x\rightarrow0} \frac{\sin(x)}{x} = 1
$$
```

提交后，配合浏览器插件 [MathJax Plugin for Github](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima) 可以渲染出数学公式。助教端已安装，可以直接渲染，大家直接在报告种插入数学公式代码即可，或者采用别的方式插入公式，比如直接贴图片也行，但需要考虑美观。

$$
\lim_{x\rightarrow 0} \frac{\sin(x)}{x} = 1
$$

## 代码实现

尽量讲清楚自己的设计，以上分析的每个技术难点分别采用什么样的算法实现的，可以是自己写的（会有加分），也可以调包。如有参考别人的实现，虽不可耻，但是要自己理解和消化，可以摆上参考链接，也鼓励大家进行优化和改进。

- 鼓励大家分拆功能，进行封装，减小耦合。每个子函数干的事情尽可能简单纯粹，方便复用和拓展，整个系统功能也简洁容易理解。
- 尽量规范地命名和注释，使代码容易理解，可以自己参考网上教程。

插入算法的伪代码或子函数代码等，能更清晰地说明自己的设计，其中，可以用 markdown 中的代码高亮插入，比如：

```python
data = ["one", "two", "three"]
for idx, val in enumerate(data):
    print(f'{idx}:{val}')

def add_number(a, b):
    return a + b
```


## 效果展示

在这儿可以展示自己基于素材实现的效果，可以贴图，如果是视频，建议转成 Gif 插入，例如：

![AR 效果展示](demo/ar.gif)

如果自己实现了好玩儿的 feature，比如有意思的交互式编辑等，可以想办法展示和凸显出来。

## 工程结构

```text
.
├── code
│   ├── run.py
│   └── utils.py
├── input
│   ├── bar.png
│   └── foo.png
└── output
    └── result.png
```

## 运行说明

在这里，建议写明依赖环境和库的具体版本号，如果是 python 可以建一个 requirements.txt，例如：

```
opencv-python==3.4
Flask==0.11.1
```

运行说明尽量列举清晰，例如：
```
pip install opencv-python
python run.py --src_path xxx.png --dst_path yyy.png
npm run make-es5 --silent
```

