# **Module 5: Semantic Segmentation**

### **一、语义分割（Semantic Segmentation）**

常见的语义分割模型为输入一幅图片、场景。输出为：图片内**每个像素的分类得分**。与目标检测不同，目标检测处理图像的一部分而语义分割处理图片的全部像素。因此在自动驾驶领域，类似于车道线识别的任务只能由语义风格模型完成。

设输入为$M*N$分辨率的图片。则语义分割模型输出:$M*N*(class_1,class_2,...,class_n)$形状的每个像素的分类得分。如下图所示：

![image-20210914140454457](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210914140454457.png)

最早的[FCN模型](https://arxiv.org/abs/1411.4038)使用**纯卷积网络**实现了场景的语义分割。CNN网络是目前语义分割模型最普遍的架构。

![image-20210914140938457](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210914140938457.png)

#### **语义分割任务的难点**

- 遮挡和截断（Occlusion，Truncation）依然是语义分割任务中的难点。这体现在人工为每个像素标注类别标签时，如果某个像素同时属于多个物体（遮挡问题）应该给该像素打上什么标签？同时同类型的物体在图片上发生遮挡时，语义分割的结果会“糊”在一起难以鉴别。如下图场景内有许多蓝色的汽车，但是全部"糊"在一起了。![img27rgb](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/img27rgb.png)

- 尺度：近大远小；光照；平滑的边界线：如马路和人行道直接过度的边缘。

#### 语义分割的性能指标

**类别IOU**：该指标针对的是某一个类别的预测结果与Ground Truth的差别

- True Positive（TP）：Ground Truth 为C类且预测也为C类的像素数量。
- False Positive（FP）：预测为C类，但是Ground Truth不是C类的像素数量。
- False Negative（FN）：预测不是C类，但是Ground Truth 为C类的像素数量。

有了以上指标后，则类别的IOU为:$IOU_{class} = \frac{TP}{TP+FP+FN}$

因此**平均IOU（MIOU）**定义为：全部类别IOU的平均值

其中注意区分FP和FN的定义，**一定要确认计算的是哪个类别的IOU**！见以下两个例子：

![image-20210914144116353](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210914144116353.png)

计算**Road的IOU**。注意两个预测为Sidewalk像素，**它们的预测值不是R**。但是Ground Truth为R。因此这两个像素应该划分为FN。

![image-20210914144725148](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210914144725148.png)

而计算**Sidewalk的IOU**时。注意两个预测为Sidewalk像素，**它们的预测值是S**。但是Ground Truth为R。因此这两个像素应该划分为FP。

**像素精度（Pixel Accuracy、PA）:**直接就是分类正确的像素占全部像素的比例

**平均像素精度（MPA）：**计算每个类别的PA，然后求平均



### **二、语义分割模型框架**

与目标检测一样，语义分割模型也要先使用CNN网络提取图片的特征图。但是常用的VGG、AlxeNet、Resnet等CNN网络输出的特征图的尺寸都会相对于原始图片**缩小k倍**。前面提到过，语义分割是在原始图片上对像素进行分类的模型。特征图的像素显然与原始图片上的像素不能一一对应。为此提出两种解决方案：

**方案一：**在特征图上进行像素分类，然后使用原始图片和特征图的**放缩系数**将特征图上的像素**上采样**到原始图片的一块$k*k$的区域。

![image-20210914153433089](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210914153433089.png)

这种方案有个致命的缺点：特征图上的像素上采样到原始图片上对应的是一个区域，因此原始图像上是**按照“块”进行分类**的。这导致语义分割的结果会比较粗糙，且原始图像上分辨率低于$k*k$的物体在特征图上没有对应的像素点，丢失了图像细节。使用这种方法输出的效果如下。

![image-20210914154044760](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210914154044760.png)

**方案二：**使用一种方法**将特征图上采样到原始图片的尺寸后**，再进行像素分类。这是语义分割的主流方案，其特征图尺寸变化同“漏斗”一样。输入、输出尺寸一样，而模型的中部特征图最小。

![image-20210914154451575](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210914154451575.png)

![image-20210914155238486](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210914155238486.png)

注：既然要保证卷积的输入和输出的尺寸一样，为什么全部使用Same 卷积保证卷积后尺寸不变？因为高分辨率的Same卷积计算资源消耗大....

#### 上采样的方法

这里直接使用CS231n课程中的笔记：

**上采样和下采样**

下采样：卷积网络中下采样的实现很简单，即maxpooling等池化操作

上采样：

图片经过pooling后，理论上是不可逆的。比如max pooling之后，图片被pooling后只会保留数值最大的区域，其他区域被丢弃了

为此提出了以下几种人为规定的上采样方式

- 最近领域插值（Nearest Neighbor）![image-20210419164925565](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210419164925565.png)

- Bed of Nails![image-20210419164955917](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210419164955917.png)

- Max Unpooling:在max pooling时记录最大值的位置，Unpooling时使用这些记录的位置进行数值插入，如下所示

  ![image-20210419165222984](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210419165222984.png)

- **双线性插值**（Bilinear Interpolation）：略

  我在Unet中使用Bilinear Interpolation方法的模型比使用反卷积（转置卷积）的方法的模型。训练参数量上少了大约一半，而效果却差不多


- **可学习的上采样方法——反卷积、转置卷积（Transpose Convolution）**

  粗劣的讲，单纯的转置卷积只能帮助恢复到原始尺寸，并不能保证恢复到原始数值。因此使用**可学习的卷积核**，希望能够学习到帮助恢复到原始尺寸的参数。

  pytorch中直接调用nn.ConvTranspose2d（）

  [具体分析参考博客](https://blog.csdn.net/tsyccnh/article/details/87357447)

  

### **三、使用语义分割的输出辅助理解道路的场景信息**

语义分割的结果可以高效的从图像中筛选出我们感兴趣的像素区域。为进一步的算法处理提供了约束空间。比如需要识别出道路上的障碍物，就可以使用分割结果中的道路类别的像素以约束问题空间。

下面介绍两个基于语义分割结果的道路场景识别任务：可行驶空间估计和车道线估计。



#### **3D可行驶空间估计（3D Drivable Space Estimation）**

可行驶空间定义为：车辆前方可以安全行驶的区域。通常由语义分割输出的像素类别中：道路、斑马线、道路上的标记、车道标记等都可以作为可行驶空间的候选区域类别。

通常3D可行驶空间估计算法有以下步骤：

- 生成场景的语义分割图
- 将3D点云投影到分割图内
- 利用投影分布在**可行驶空间候选区域类别**中的点云数据**估计出可行驶空间表面的表达式**

**3D可行驶平面模型**

定义一个最简单的**平面模型**:$p:ax+by+z=d$，其中$(x,y,z)$。表点云中在该平面上的点的坐标，而$(a,b,d)$是该模型的待求解参数。$z$的系数为1代表法向量，$d$为偏移常量。

因此给定N个点在语义分割图上的坐标$(x_i,y_i)$，以及它们的深度$z_i$

需要找到3个参数$(a,b,d)$确定平面$ax+by+z=d$

在这个平面方程中带入$(x_i,y_i)$，即$-z=ax_i+by_i-d$

最后最优平面应该保证$-z$和$z_i$之间的差距最小，即求:

$argmin(-z-z_i)=argmin(ax_i+by_i-d-z_i)$

因此令平面模型为$p=[a,b,d]$,N个点在语义分割图上的坐标矩阵为A,深度矩阵为B。A、B如下所示：
$$
A=
\begin{bmatrix}
x_1&y_1&-1\\
x_1&y_1&-1\\
\vdots&\vdots&\vdots\\
x_n & y_n&-1
\end{bmatrix}
B=
\begin{bmatrix}
-z_1\\
-z_2\\
\vdots\\
-z_n
\end{bmatrix}
$$
最终需要求解最优问题：$\mathop {argmin}_{A}(Ap-B)$

上式有最小二乘解：$p=(A^TA)^{-1}A^TB$

![image-20210915141733812](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210915141733812.png)

**使用RANSAC解可行驶平面模型**

在这个模型中需要确定的参数有3个，即$p=[a,b,d]$。因此使用RANSAC算法即该模型，也需要**至少3个不共线**的的点。由此可以执行以下的RANSAC算法流程：

- 随机选择分割图内的3个投影点

- 使用这3个点和最小二乘法求解$p$
- 使用当前$p$确定的模型，和最小阈值。确定剩下的点有多少是内点（符合模型的点），并暂存这些内点点集。
- 内点数量大于预先设定好的阈值。就终止循环，否则返回第一步继续迭代。
- 退出循环时，使用最终暂存的内点点集，求解最终的模型参数$p$



#### **车道估计（Lane Estimation）**

估计出了可行驶空间，还需要告诉车辆可以在这个空间内的哪些位置行驶。因为车辆需要在车道标记和路的边界线内的空间行驶。为此需要估计车道。

车道估计的基础是，找出车道的边界。语义分割输出的标签中，可能用于估计车道边界的类别有。路肩(马路牙子）、道路、路边停靠的车辆。

![image-20210915150244873](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210915150244873.png)

以下为车道估计的常见流程：

- 从语义分割的输出类别中，筛选出能用于车道划分的像素。如路标、路肩...用二值掩码图像表示筛选结果。（被筛选的类别的像素值为255其他为0）
- 使用边缘检测器（edge detector），如Canny edge detector。提取二值掩码图像内的边缘
- 使用一个车道模型，比如直线车道模型(Linear Lane Model)结合检测到的线段来估计最终的车道线模型。其中“线段”数据可以使用对齐检测器（aligned detector），如Hough Transform line detector 从"边缘"数据找出的线段。
- 筛选上一步获得的“线段”数据集。比如去除水平的线段，去除可行驶平面之外的线段
- 最后需要确定车道两侧的部分的类别

#### 最终编程练习：可行驶空间估计、车道线估计、目标距离估计

见assignment/M3&M4&M5

![image-20211005164723616](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/NZfC3cxSkbo5LvJ.png)

![image-20211005164734366](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/vwFUgWu3nam5OJM.png)

![image-20211005164753195](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/UL6wp32c1NEHJe8.png)

