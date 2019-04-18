**论文传送门：**

[Drop an Octave: Reducing Spatial Redundancy in Convolutional Neural Networks with Octave Convolution](https://export.arxiv.org/pdf/1904.05049)

## 1. Motivation

声音频段中有高频部分和低频部分，图片像素中也可以将信息分为高频信息和低频信息，低频信息中包含那些缓慢变化的、结构性的信息，而高频信息一般包含那些变化较大、包含图片细节的信息，因此我们可以把一张图片认为是高频信息和低频信息的混合表示。下图是一个图像的高频信息和低频信息分离表示。

<center>
<img src="https://github.com/serryuer/blog-img/raw/master/imgs/Snipaste_2019-04-18_16-09-53.jpg" width=100% height=100%>


为了提高CNN的性能和准确率，有很多的工作致力于减少**模型参数**内在的冗余，在加快训练速度的同时能够获得更多正交的高级特征从而提高模型的表示能力，但是实际上每一层CNN输出的**特征图**也存在着大量的冗余，我们以原始输入为例，图片的每一个像素之间并不是孤立的，相邻的元素之间存在着密切的联系，将它们联合在一起存储表示往往能够更加准确的描述图片的信息，所以我们以单一像素的形式存储是及其浪费存储和计算资源的。

因此，我们可以将特征图按照频率分成两部分，一部分是高频信息，一部分是低频信息，然后我们可以将低频信息的数据通过使相邻元素之间共享信息来降维，进而降低其空间冗余性。在对低维数据进行处理的时候，我们还相当于变相的扩大了其感受野，使得网络能够学习到更加高级的特征，有助于提高网络的准确性。

本文的贡献如下：

1. 提出将卷积特征图按照不同的频率分成两组，并对不同频率的数据进行不同的卷积处理。因为低频信息可以被安全的压缩而不用担心影响网络的准确率，所以这可以使得我们节省大量的存储和计算资源；
2. 设计了一个名为OctConv的即插即用的运行单元，可以直接作为普通CNN卷积运算单元的替换，OctConv可以直接处理我们提出的新的特征图表示并且降低了低频数据的冗余性；
3. 使用上面设计的OctConv实现流行的CNN框架，并与当前最好结果进行对比，同时和当前最好的AutoML框架进行了对比。

##  2. 相关工作

### 2.1 提高CNN框架的效率

自从AlexNet、VGG等框架使用堆叠大量卷积层的方法构架网络并取得了巨大的成绩之后，研究者们为了提高CNN的效率做出了很多的努力。

ResNet、DenseNet通过增加跨越多层的快捷连接来加强特征的重用，同时也是为了减轻梯度消失问题，降低了优化的困难。如下图所示：

<center>
<img src="https://github.com/serryuer/blog-img/raw/master/imgs/Snipaste_2019-04-18_16-04-43.jpg" >

因为我们可以在现有网络结构上直接堆叠一个恒等映射层，也就是一个什么都不做的层，而不影响整体的效果，所以深层网络不应该比稍浅一些的网络造成更大的误差，所以ResNet引入了残差块，并且希望该结构能够fit剩余映射，而不是直接去fit底层映射。ResNet并不是第一个利用快捷连接的神经网络，Highway Network首次引入了Gated shortcut connections，用于控制通过该连接的信息量，同样的思想我们还可以在LSTM的设计中发现，它利用不同的门控单元来控制信息的流动。
ResNeXt和ShuffleNet框架利用稀疏连接的分组卷积方法来降低不同通道之间的冗余性，如下图所示：

<center>
<img src="https://github.com/serryuer/blog-img/raw/master/imgs/Snipaste_2019-04-18_16-05-02.jpg">

Xception和MobileNet使用深度可分离卷积层来减小连接的密集程度。

### 2.2 多尺度表示学习
本文提出的OctConv在不同的空间分辨率下对特征图进行卷积，得到了具有较大接收域的多尺度特征表示。尺度空间很早就被用于局部特征提取，比如之前很流行的SIFT特征，在深度学习领域，现有方法主要聚焦于融合多尺度特征，更好的获取全局信息。然而类似的这些方法大多只在网络结构的某些曾或者网络的末端加入提出的新的结构，bL-Net和ELASTIC-Net使用频繁的对特征图进行上采样和下采样来捕获多尺度特征。但是以上这些方法都是设计为残差块的替代品，当被用于不同的网络结构时需要额外的专业知识和更多的超参数优化。

## 3 网络结构

### 3.1 Octave特征表示

对于一般的CNN来说，输入和输出特征图的空间分辨率是相同的，然而对于空间频率模型，即本文提出的方法来说，一个自然图片可以被因式分解成捕获全局信息和粗略架构的低频信号和捕获优化细节的高频信号。我们认为存在一个特征图的子集可以空间中的低频信号的变化和冗余信息。

为了减少这种空间冗余性，我们引入了Octave特征表示，显示的将特征图分成了高频和低频两部分。假设输入特征图为$X \in R^{c*h*w}$，其中c是通道数，h、w是特征图的尺寸，我们在通道维度将特征图分为两部分，即$X=\{X^H, X^L\}$，其中，$X^H \in R^{(1-\alpha)c*h*w}，X^L \in R^{\alpha c * \frac{h} {2}* \frac{w}{2}}, \ \ \alpha \in \{0,1\}$，$\alpha$表示通道被分配到低频部分的比例。如下图所示：

<center>
<img src="https://github.com/serryuer/blog-img/raw/master/imgs/Snipaste_2019-04-18_16-05-09.jpg" width=50% height=50%>

### 3.2 Octave 卷积
上面介绍的Octave特征表示降低了低频信息的空间冗余性，并且比正常CNN特征表示更加复杂，所以不能直接被正常的CNN所处理，一直比较Naive的做法是对低频表示进行上采样使其达到和高频信息一致的尺寸然后组合这两个部分，就可以得到能够被普通CNN处理的数据格式，但是这种做法即增加了内存的消耗，也增加了大量的计算。因此，本文设计了可以直接处理$X=\{X^H, X^L\}$这种特征表示的卷积层：Octave  Convolution。

**Vanilla Convolution**：
设$W \in R^{c*k*k}$表示一个卷积核，$X,Y \in R^{c*h*w}$表示输入和输出张量，那么输出特征$Y_{p,q} \in R^c$可以表示为：
$$Y_{p,q}=\sum_{i, j  \in N_kW_{i + \frac{k-1}{2}, j + \frac{k-1}{2}}^T X_{p+i, q+j}$$
上式比较简单，这里不多做解释。

**Octave Convolution**：
我们的目标是在分别对高频和低频信息进行卷积处理的同时还允许二者之间进行有效的信息交互。因此由上面的介绍我们可以得到：
$$
Y^H=Y^{H\rightarrow H} + Y^{L \rightarrow H}\\
Y^L=Y^{L\rightarrow L} + Y^{H \rightarrow L}\\
$$
如下图所示：

<center>
<img src="https://github.com/serryuer/blog-img/raw/master/imgs/Snipaste_2019-04-18_13-24-44.jpg" width=50%, height=50%>

为了上面的计算，我们将卷积核W分成四个部分$W=\{W^H, W^L\}=\{W^{H\rightarrow H}, W^{H\rightarrow L}, W^{L\rightarrow L}, W^{L\rightarrow H}\}$，如下图所示：

<center>
<img src="https://github.com/serryuer/blog-img/raw/master/imgs/Snipaste_2019-04-18_16-05-30.jpg" width=50%, height=50%>

对于频内信息交流，我们直接使用普通的CNN卷积计算，对于频间的信息交流，我们将上/下采样和卷积操作放在一起，避免了显示的计算和存储采样结果。公式如下：
$$
\begin{split}
Y_{p,q}^H&=Y_{p,q}^{H\rightarrow H} + Y_{p, q}^{L\rightarrow H}\\
&=\sum_{i,j \in N_k}{W_{i+\frac{k-1}{2}, j +\frac{k-1}{2}}^{H\rightarrow H}}^T X_{p+i, q+j}^H\\
&+\sum_{i,j \in N_k}{W_{i+\frac{k-1}{2}, j +\frac{k-1}{2}}^{L\rightarrow H}}^TX_{\lfloor \frac{p}{2} \rfloor +i, \lfloor \frac{q}{2} \rfloor+j}^L
\end{split}
$$
$$
\begin{split}
Y_{p,q}^L&=Y_{p,q}^{L\rightarrow L} + Y_{p, q}^{H\rightarrow L}\\
&=\sum_{i,j \in N_k}{W_{i+\frac{k-1}{2}, j +\frac{k-1}{2}}^{L\rightarrow L}}^T X_{p+i, q+j}^L\\
&+\sum_{i,j \in N_k}{W_{i+\frac{k-1}{2}, j +\frac{k-1}{2}}^{H\rightarrow L}}^TX_{2*p+0.5+i, 2*q+0.5+j}^H
\end{split}
$$

上式中的$\lfloor \frac{p}{2} \rfloor$是为了上采样，同样的，$2*p+0.5+i$的目的是下采样，加上0.5度目的是为了保证下采样之后的输出和输入的尺寸是一一致的，相当于平均池化。

### 3.3实现细节
待续