# Visual Perception for Self-Driving Cars

参考视频：[Coursera](https://www.coursera.org/learn/visual-perception-self-driving-cars?specialization=self-driving-cars)

Note、assignment和ppt :[Github](https://github.com/phww/self-driving-car)

## **Module 1: Basics of 3D Computer Vision**

### 1. 小孔成像（Pine Hole Model）

小孔成像模型有两个重要的参数：**焦距（focal length）和小孔中心坐标（camera center）**。其中**焦距决定了成像的大小**，而小孔中心坐标可以帮助人们确定物体在图像上的**映射信息**。有了这两种参数就可以使用数学上的方法，确定物体在底片上的成像位置和描述图像的生成过程。进而为状态估计和目标检测等任务提供帮助。



### 2. Camera Projective Geometry（相机投影几何）

#### 建模：将真实世界的物体投影到相机的底片上

![image-20210810185223779](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210810185223779.png)

**问题**：如果称人为规定的物体所在的真实世界的坐标系为“世界”坐标，而物体在成像平面上的投影图像的坐标系为图片坐标。现在假设在世界坐标系下的某个位置$(x_c,y_c,z_c)$有一个相机镜头，求物体在相机镜头后的成像平面上的坐标。

因为小孔成像得到的图片是上下颠倒的。为了防止混淆，人为设定一个**虚拟的成像平面**代替原始的成像平面。就可以得到下图所示的相机投影模型。问题便转换为如何将世界坐标系上的点(X, Y, Z)投影到虚拟成像平面以(u，v)表示。

![image-20210810190148239](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210810191807061.png)

其中：

- 首先需要定义**世界坐标系（word coordinate）**。定义相机镜头在世界坐标系下的坐标即相机的位姿(pose)，以这个坐标为中心定义**相机坐标系$X_cY_cZ_c$（camera coordinate）**
-  之后定义成像平面$X_sY_xZ_s$（image coordinate）的坐标原点，为虚拟成像平面的中心。而虚拟成像平面的原点位于其左上角。为了做区分，下面称虚拟成像平面为**像素坐标系（pixel coordinate）**
- 定义相机坐标系和成像平面相对于Z轴的距离为**焦距f**

**投影方法：**

- 使用旋转矩阵、位移矩阵[R|t]，将世界坐标系转换为相机坐标系

  ![image-20210810194452243](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210810194452243.png)

- 使用尺度变换（scale）和偏移变换（offset）的矩阵K，将相机坐标系转换为成像坐标系。K矩阵也称为相机的**内参**，是相机自身确定的参数

  ![image-20210810194526496](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210810194526496.png)

- 因此只要定义矩阵[R|t]和矩阵K，则通过矩阵乘法可以得到世界坐标系到成像坐标系的转换矩阵$P=K @ [R|t]$

  ![image-20210810194555212](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210810194555212.png)

- 使用**齐次坐标**和转换矩阵P可以将世界坐标$(X,Y,Z,1)^T$投影到成像坐标$(x,y,z,1)$。之后将x、y坐标除以z，最终得到像素坐标系下的坐标$(u,v)$其中$u=\frac{x}{z}, v=\frac{y}{z}$

  ![image-20210810194638191](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210810194638191.png)

- 最后在实际应用中，还会遇到相机失真、像素横纵比失衡等情况。但是只需要调整矩阵K就可以解决这些问题。

**最终成像：**

上面的坐标系系转换只是将物体的坐标从世界坐标系投影到了像素坐标系。设图片大小为MxN，有了(u,v)坐标之后还需要在对应的坐标网格中填入像素值。如果只是填入0-255单通道像素，那么得到的图像就是灰度图。彩色图像还需要对3个MxN矩阵，即R、G、B通道进行像素填充

![image-20210810194818651](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210810194818651.png)

注：标准内参矩阵$K$和外参矩阵$[R|t]$如下所示：
$$
K=
\begin{bmatrix}
f_x&0&x_0\\
0&f_y&y_0\\
0&0&1
\end{bmatrix}
&&&&&
[R|t]=
\begin{bmatrix}
R&t\\
0_3^T&1\\
\end{bmatrix}
$$
内参矩阵中：

- $f_x,f_y$表示相机的焦距。理想情况下，小孔成像的相机$f_x=f_y$

- $(x_0,y_0)$表示主点偏移。主点是相机镜头位置投影到成像平面上的点位置，该点在图像坐标系下的坐标就是主点偏移量。如下所示：

  ![image-20210916153843925](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210916153843925.png)

外参矩阵中：

- 旋转矩阵$R$为3x3的矩阵。比如如绕X轴旋转$\theta$度的旋转矩阵为：
  $$
  R=
  \begin{bmatrix}
  1&0&x_0\\
  0&cos\theta&sin\theta\\
  0&-sin\theta&cos\theta
  \end{bmatrix}
  $$

- 偏移向量为3x1的矩阵。即$[t_x,t_y,t_z]^T$分别代表相对于每个坐标轴的偏移量

### 3. Camera Calibration(相机校正)

相机校正的目的就是获得：相机的**内外参数**，即内参矩阵K和相机的位姿矩阵[R|t]。而最基础的单相机校准的方法是通过人为测量一组世界坐标系下的点，和与其对应的像素坐标系下的点。先逆向**求转换矩阵P**，再**使用QR分解矩阵P**得到**内参矩阵K和位姿矩阵[R|t]**。如下所示，s表示伸缩比例（scale）。

![image-20210811191800488](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210811191800488.png)

- 求解转换矩阵P

  ![image-20210812191010761](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210812191010761.png)

  使用如上的check board，可以准确的从2d点坐标获取3d点坐标。然后根据两者的对应关系，求解参数方程。首先，列出方程组

  ![image-20210812191350747](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210812191350747.png)

  之后将等式（3）带入等式（1）（2）中，得到：

  ![image-20210812191459391](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210812191459391.png)

  使用矩阵表示以上结果：

  ![image-20210812191533001](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210812191533001.png)

  如果选取了**N个点**进行校正，那么将会得到**2*N个齐次线性方程组**。解这样的齐次线性方程组可以使用求广义逆矩阵、奇异值分解等方法获得最小二乘解。

- 分解矩阵P得到相机的内参矩阵K和外参（位姿）矩阵[R|t]

  首先，如果相机中心在世界坐标系下的坐标向量为C。则将其投影到相机的成像坐标系时，这个点刚好是成像坐标系的零点。因此PC=0.所以有以下公式：

  ![image-20210812193322892](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210812193322892.png)

  得出平移矩阵t=-RC后，带入原式$P=K @ [R|t]$中有：

  ![](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210812193533635.png)

  令矩阵M=KR，则$P=[M|-MC]$​，其中M是一个方阵。根据矩阵论，任何一个n阶方阵都可以分解为n阶的上三角矩阵和n阶正交基的乘积。因此M=$R$​Q=KR，其中**$R$​为上三角矩阵**，**Q为正交基**。进一步可以认为，**上三角矩阵$R$​和内参矩阵K相等。而正交基Q和旋转矩阵R相等**。而平移矩阵t=-RC，M=KR->R=$K^{-1}M$​。则t=$K^{-1}MC$​=$K^{-1}P[:,4]$​​​。转换过程如下：
  
  ![image-20210812194828899](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210812194828899.png)

#### **编程练习：给出相机的投影矩阵$P$，分解投影矩阵获取相机的内外参矩阵$K$和$[R|t]$**

opencv中直接使用[cv2.decompose_projection_matrix](https://docs.opencv.org/3.4.3/d9/d0c/group__calib3d.html#gaaae5a7899faa1ffdf268cd9088940248)

numpy提供的[QR分解](https://numpy.org/doc/stable/reference/generated/numpy.linalg.qr.html)也可以得到同样的结果

```python
def decompose_projection_matrix(p):
    k,r,t, _,_,_,_ = cv2.decomposeProjectionMatrix(p)
    t /= t[3]
    return k, r, t
def decompose_projection_matrix_v2(p):
    Q,R = np.linalg.qr(p)
    k = R[:,:-1]
    r = Q
    t = -np.linalg.pinv(k)@p[:,-1]
    return k, r, t
```



### 4.Stereopsis（立体视觉）

通常只有深度摄像头和雷达等传感器才能直接收集到物体的深度信息。而普通相机想要获取物体的深度信息，就需要使用多视角（>=2）的立体视觉的方法，通过数学计算得到物体的深度信息。

**双目立体视觉模型**

- 如下所示，点O在两个**平行的相机**的成像平面上的投影$O_L,O_R$。两个相机之间在x轴上的**偏移为b**，成像的**焦距为f**。立体视觉模型的目的就是计算点O在相机坐标系下的深度Z。

  ![image-20210815170627241](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210815170627241.png)

  其中最基础的双目立体视觉模型需要满足以下条件

  - 使用两个**内参相同**的相机
  - 且两个相机的成像平面的**x轴是对齐的**，只是两个相机坐标系下的x轴有相对的偏移d。

  因此从世界坐标系到两个相机坐标系的**旋转矩阵R应该是一样的**。而两个相机坐标系相互转换也只需要一个关于x轴相对偏移距离b的偏移矩阵。最终从世界坐标系到这个“双目立体视觉模型”坐标系的变化由**旋转矩阵R,偏移矩阵t**以及两个相机之间在x轴上的**偏移b**决定。

- **计算物体的深度Z**

  ![image-20210815171118148](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210815171118148.png)

将上面的双目立体视觉模型投影到鸟瞰图，很容易根据两组**相似三角形**推断出以下的式子：
$$
\frac{Z}{f}=\frac{X}{x_L},\frac{Z}{f}=\frac{X-b}{x_R}\\
$$
根据以上公式，定义**视差（disparity）**$d=x_L-x_R$。则可以推算出物体的**深度$Z=\frac{fb}{d}$**。如下图所示：

![image-20210815172731372](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210815172731372.png)

因此为了最后计算出深度Z。需要知道相机的**焦距f**、两个相机在x轴上的**偏移b**以及同一个点在两个成像平面上的**视差d**。其中**f和b可以根据相机的内外参获取。**

**获取视差**

获取视差的方法就是找到同一个点在两个成像平面上的投影位置$(x_L,y)$和$(x_R,y)$（point pairs）。一个较为简单的方法就是先用双目立体视觉系统拍摄两张图片，然后固定图片的y坐标。遍历两张图片上一条直线上的所有像素点，寻找**相似度最高**的点对。最后计算出该立体视觉系统的视差d。

![image-20210815173725412](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210815174925161.png)

但是上面的方法需要一个前提：两个图片的**极线（epipolar plane）需要平行**。

![image-20210815174925161](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210815174407486.png)

​	如果不平行，则无法沿着同一y坐标进行遍历寻找点对

![image-20210815174407486](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210815173725412.png)

当然，可以使用**立体矫正（stereo rectification）**的方法解决这一问题。

![image-20210815175551447](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210815175551447.png)

**总结**

通过立体视觉算法计算物体的深度的基础方法：

![image-20210815180117891](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210815180117891.png)

最后根据公式$Z=\frac{fb}{d}$计算出图像上每个像素的深度信息。

#### **编程练习：给定同一场景的两个视角的图片：lmg_left, img_right。和相机的内参矩阵。求左视角图片的深度图**

示例图片如下：

![image-20210916163939205](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210916163939205.png)

**获取视差图：**使用[cv2.SteroBM](https://docs.opencv.org/3.4.3/d9/dba/classcv_1_1StereoBM.html)或[cv2.StereoSGBM](https://docs.opencv.org/3.4.3/d2/d85/classcv_1_1StereoSGBM.html)。前者参数只有两个。官方只有这个练习给了标准答案，是因为参数太难调了吗...

```python
def compute_left_disparity_map(img_left, img_right):
    # 超参
    num_disparities = 6*16
    block_size = 11
    min_disparity = 0
    window_size = 6
    # 计算视差图，要求输入灰度图
    img_left = cv2.cvtColor(img_left, cv2.COLOR_BGR2GRAY)
    img_right = cv2.cvtColor(img_right, cv2.COLOR_BGR2GRAY)
    # bm
    left_matcher_BM = cv2.StereoBM_create(
        numDisparities=num_disparities,
        blockSize=block_size
    )
    #sgbm
    left_matcher_SGBM = cv2.StereoSGBM_create(
        minDisparity=min_disparity,
        numDisparities=num_disparities,
        blockSize=block_size,
        P1=8 * 3 * window_size ** 2,
        P2=32 * 3 * window_size ** 2,
        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY
    )
    disp_left = left_matcher_SGBM.compute(img_left, img_right).astype(np.float32)/16
    # disp_left = left_matcher_BM.compute(img_left, img_right)
    return disp_left
```

输出结果为与**原始图片大小一样的视差图矩阵**：注意视差图的左边有一小段的像素全部被填充为-1。猜测是因为左视角最左边的部分在右视角的图片上没有对应的部分，无法计算视差。因此填充为-1。

不要尝试用StereoSGBM或StereoBM去计算右视角图像的视差图。因为这两个算法都是基于算左视差图建模的。计算右视差图得不到正确结果。

![image-20210916164109153](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210916164109153.png)

计算深度：

```python
def calc_depth_map(disp_left, k_left, t_left, t_right):
    # 焦距和左右相机偏移量
    f = k_left[0][0]
    b = t_left[1] - t_right[1]
    # 视差图中有为0和-1的元素
    disp_left[disp_left == 0] = 0.1
    disp_left[disp_left == -1] = 0.1
    # depth = f * b / d
    depth_map = f*b / disp_left
    return depth_map
```

![image-20210916165617637](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210916165617637.png)



### 5.图像滤波（filter）、互相关和卷积（ Cross Correlation/ Convolution）

图像滤波有许多内容，这里只记录原理。

**filter kernel**

定义filter kernel为NxN的窗口。将这个窗口在图片上滑动，并按照一定的规则计算窗口内的像素。公式如下：$G[u,v]=HI[u-i][v-j]，其中-N<=i、j<=N$。简记为$G=HI$

其中u,v对应窗口内的中心像素的坐标，窗口内其他像素的坐标用[u-i]\[v-j]表示，注意i和j的取值范围。因此G[u,v]代表原始图像中以[u,v]像素为中心周围NxN范围内的像素经过特定的滤波操作F后的输出。

如下图，这里的滤波操作F:使用窗口内所有像素的均值作为中心[u,v]像素的输出。使用这种滤波操作，可以有效的去除图片的椒盐噪声。

![image-20210819171232387](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210819171232387.png)

**互相关和卷积**

**互相关操作定义为**：在filter kernel中填充NxN的数值，然后定义滤波操作H为$G[u,v]=\sum_{i=-k} ^{k}\sum_{j=-k}^kH[i][j]I[u-i][v-j]$，简记为$G=H\bigotimes I$。可以理解为对应窗口内 kernel 和图像像素对应位置的元素相乘在相加得到的数值作为G[u,v]的输出，如下所示：

![image-20210819172331181](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210819172331181.png)

**卷积的定义为**:将kernel先水平和垂直翻转（即沿着kernel的主对角线将kernel对应的位置的元素交换），然后再使用互相关的滤波操作，简记为简记为$G=H\ast I$

![image-20210819173141191](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210819173141191.png)

如过定义翻转操作为F，则卷积计算公式可表示为$H*(F*I)$。因为卷积是**相关联的（associative？？）**，则$(H*F)*I$。定义新的滤波操作为$\hat H=H*F$，则卷积操作的公式依然是$G=H*I$

注：当filter kernel上下左右对称时，卷积和互相关的结果就没有区别了

**应用**

- 针对不同的噪点类型使用各种对应的filter，对图片进行降噪。

- 使用互相关，进行摸板匹配。将模板与图片做互相关后，输出G[u,v]中元素最大的位置[u,v]就是摸板在图片中匹配位置的中心。

  ![image-20210819191629668](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210819191629668.png)

- 使用各种卷积核提取图片的特征，因为卷积的相关性。因此可以将提取多种特征的卷积核依次与原始图片做卷积操作，最终得到更加抽象的特征。

  ![image-20210819191648255](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210819191648255.png)

**CNN中的迷惑**

​	之前学习CNN时，总结的卷积操作就是窗口内卷积核和图片对应位置的元素相乘再相加（加权求和）。严格来说这是互相关操作。但是CNN中由于是卷积核自学习核内的参数，因此严格使用互相关或卷积操作训练出来的卷积核，最后的差别应该就是参数的位置顺序。因此CNN中严格使用互相关或卷积的差别不大，严格使用自相关还能省下严格卷积的旋转操作。

但是在图像处理领域中一定要区分这两种概念！！有一些中文教程，没有区分这两种概念，就很迷惑。。。

#### **编程练习：模板匹配**

测试的图片为深度估计练习中的左视角图片，用于匹配的摸板如下：

![image-20210916172918330](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210916172918330.png)

[cv2.matchTemplate](https://docs.opencv.org/3.4.3/df/dfb/group__imgproc__object.html#ga586ebfb0a7fb604b35a23d85391329be)输入原始图像和模板图像，会输出每个窗口的计算结果矩阵。设输入图片尺寸为$W,H$,模板图像为$w,h$。则输出矩阵尺寸为$(W-w+1,H-h+1)$（就是滑动窗口的数量）

其中函数提供了几种窗口内的计算方式。使用互相关，method设置为TM_CCOEFF或TM_CCOEFF_NORMAL。也只有这两种方式能通过官方的编程练习...

![image-20210916171929027](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210916171929027.png)

实际应用中更关心模板的位置，即matchTemplate输出中最大值的位置。使用[cv2.minMaxLoc](https://docs.opencv.org/3.4.3/d2/de8/group__core__array.html#ga8873b86a29c5af51cafdcee82f8150a7)找出最大和最小值的位置。（这里只需要最大值）

```python
def locate_obstacle_in_image(image, obstacle_image):
    cross_corr_map = cv2.matchTemplate(image, obstacle_image, cv2.TM_CCOEFF)
    _,_,_,obstacle_location = cv2.minMaxLoc(cross_corr_map)
    return cross_corr_map, obstacle_location
```

![image-20210916172712296](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210916172712296.png)

根据模板图片的大小和位置，可以在左视角图片上定位模板目标。也就是用模板匹配的方法做目标识别

![image-20210916173050856](https://pic-1305686174.cos.ap-nanjing.myqcloud.com/image-20210916173050856.png)

