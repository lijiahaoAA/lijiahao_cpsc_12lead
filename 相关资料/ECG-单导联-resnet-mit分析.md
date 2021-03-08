## 单导联的分类

### 1.数据预处理

#### 1.1数据源

使用的数据库为MIT-BIH的数据。 数据的采样率为360Hz，每条记录的导联数为二导联。训练集和测试集采用了最常用的DS1（22条记录）-DS2（22条记录）inter-patient（ 对不同的病人（records）进行特征提取和分类）划分。 

 ![img](https://img-blog.csdn.net/20170620141956261) 

##### 1.1.1数据处理

MIT-BIH提供的每一条ECG数据有三个文件，分别是hea文件、dat文件、atr文件。

- hea文件：说明文件，以100.hea文件为例，内容为：

  ~~~txt
  100 2 360 650000
  100.dat 212 200 11 1024 995 -22131 0 MLII
  100.dat 212 200 11 1024 1011 20052 0 V5
  # 69 M 1085 1629 x1
  # Aldomet, Inderal
  ~~~

  所有数据表示的意义：

  ~~~txt
  100(文件名) 2(通道数/导联数 MLII和V5两路导联信号组成) 360(采样率是360Hz) 650000(每个通道信号长度为650000个采样点)
  100.dat(信号存储在100.dat文件中) 
  212(信号以212格式存储，针对两个信号的数据库记录，每三个字节(24bit=6个16进制的数字)存储两个数据(两路信号分别占一个)) 
  200(表示每个信号的增益都是每200ADC units/mV) 
  11(ADC的分辨率是11位) 
  1024(ADC的零值为1024) 
  995/1011(不同信号的第一采样点的值)  # 实际值为995/200=4.975mV
  -22131/20052(65万个采样点的校验数) 
  0(表示输入和输出可以以任意尺寸的块来执行) 
  MLII/V5(信号描述字段，表示信号采自那个导联)
  ~~~

- dat文件：数据文件，以100.dat文件为例子

  有hea文件可知，dat存储的是两个导联的电压值，存储格式为212格式，每次读取三个字节（24bit），第二个字节的低四位和第一个字节（顺序不能乱）构成第一个导联的采样电压值，第二个字节的高四位和第三个字节构成第二个导联的采样电压值。

  ![1611658281471](C:\Users\13018\AppData\Roaming\Typora\typora-user-images\1611658281471.png)

  读取0xE333F3，则第一个导联电压值为0x3E3，转化为十进制为995；第二个导联电压值为0x3F3，转化为10进制为1101.实际电压值分别为995/200=4.975mV，1101/200=5.055mV。后面以此类推。

- atr文件：注释文件，给出每个心拍的标签。以100.atr文件为例

  - 存储格式为MIT格式：![1611659643245](C:\Users\13018\AppData\Roaming\Typora\typora-user-images\1611659643245.png)

    由第一个字节不为零可知存储格式为MIT格式，读取16位值0x7012，其高六位为0x1C（十进制28），低十位的值为0x12（十进制为18），表示的意义为该心拍的诊断类型为28，发生异常的时间在18/360=0.05s；接着读出后面的16位值0xFC03，其高6为的值为0x3F（十进制63），低10位的值为0x03（十进制3），该类型代码为63，代表的意义是在该16位值后附加了3个字节的辅助信息，若字节个数为奇数，则再附加一个字节的空值，在本例中就是“28 4E 00 00”；然后再从下一字节读16位值0x043B，其高6位的值为1，低10位的值为0x3B（十进制59），该类型码1代表**正常心搏**，发生时间为0.214秒（(18+59)/360Hz）；依次类推即可读出所有的注释，当读到的16位值为0时，就表示到了文件尾。 当高6位为十进制的59时，读取之后的第三个16位的高6位作为类型代码，读取之后第二个16位+第一个16位*2^16，作为发生时间。高6位为十进制的60，61，62时继续读取下一个16位。

    下表为所有的类型代码和表示的意义：

    | 注释代码 | Symbol | Meaning                                    | 中文含义               |
    | -------- | ------ | ------------------------------------------ | ---------------------- |
    | 0        |        | No TQRS                                    |                        |
    | 1        | · or N | Normal beat                                | 正常搏动               |
    | 2        | L      | Left bundle branch block beat              | 左束支传导阻滞         |
    | 3        | R      | Right bundle branch block beat             | 右束支传导阻滞         |
    | 4        | a      | Aberrated atrial premature beat            | 异常房性早搏           |
    | 5        | V      | Premature ventricular contraction          | 室性早搏               |
    | 6        | F      | Fusion of ventricular and normal beat      | 心室融合心跳           |
    | 7        | J      | Nodal (junctional) premature beat          | 交界性早搏             |
    | 8        | A      | Atrial premature beat                      | 房性早搏               |
    | 9        | S      | Premature or ectopic supraventricular beat | 室上性早搏或异位性搏动 |
    | 10       | E      | Ventricular escape beat                    | 室性逸搏               |
    | 11       | j      | Nodal (junctional) escape beat             | 交界性逸搏             |
    | 12       | /      | Paced beat                                 | 起博心跳               |
    | 13       | Q      | Unclassifiable beat                        | 未分类心跳             |
    | 14       | ~      | signal quality change                      |                        |
    | 16       | I      | isolated QRS-like artifact                 |                        |
    | 18       | s      | ST change                                  |                        |
    | 19       | T      | T-wave change                              |                        |
    | 20       | *      | systole                                    |                        |
    | 21       | D      | diastole                                   |                        |
    | 22       | "      | comment annotation                         |                        |
    | 23       | =      | measurement annotation                     |                        |
    | 24       | p      | P-wave peak                                |                        |
    | 25       | B      | left or right bundle branch block          |                        |
    | 26       | ^      | non-conducted pacer spike                  |                        |
    | 27       | t      | T-wave peak                                |                        |
    | 28       | +      | rhythm change                              |                        |
    | 29       | u      | U-wave peak                                |                        |
    | 30       | ?      | learning                                   |                        |
    | 31       | !      | ventricular flutter wave                   |                        |
    | 32       | [      | start of ventricular flutter/fibrillation  |                        |
    | 33       | ]      | end of ventricular flutter/fibrillation    |                        |
    | 34       | e      | atrial escape beat                         | 房性逸搏               |
    | 35       | n      | supraventricular escape beat               | 室上性逸搏             |
    | 36       |        | link to external data (aux contains URL)   |                        |
    | 37       | x      | non-conducted P-wave (blocked APB)         |                        |
    | 38       | f      | fusion of paced and normal beat            |                        |
    | 39       | (      | waveform onset                             |                        |
    | 40       | )      | waveform end                               |                        |
    | 41       | r      | R-on-T premature ventricular contraction   |                        |

  - 存储格式为AHA：即采用WFDB转换的AHA数据库atr注释，第一个字节为0，读取方式和MIT格式一致.

##### 1.1.2小波变换分解与重构处理

小波系数是信号在做小波分解时所选择的小波函数空间的投影。

 一个信号可以分解为傅里叶级数，即一组三角函数之和，而傅里叶变换对应于傅里叶级数的系数；同样，**一个信号可以表示为一组小波基函数之和，小波变换系数就对应于这组小波基函数的系数。**

（1） **小波重构函数的结果都是信号；**

（2） **不管是用哪个重构函数对系数进行重构后，结果的长度和原始信号的长度是相同的；**

​     **如果重构的是低频部分，那么观察得到的结果X，其数值大小和原始信号是差不多的。**

db后面那个数字代表的是消失矩。一般来说，这个消失矩的数字越大，这个小波越光滑（长的小波滤波器）。小波滤波器的长度（尺度）是这个数字的两倍。

![1611716800550](C:\Users\13018\AppData\Roaming\Typora\typora-user-images\1611716800550.png)

~~~python
def sig_wt_filt(sig):
    """
    对信号进行小波变换滤波
    :param sig: 输入信号，1-d array
    :return: 小波滤波后的信号，1-d array

    """
    print("sig.shape={}".format(sig.shape))
    coeffs = pywt.wavedec(sig, 'db6', level=5) # 多尺度小波一维分解，5阶小波分解和2倍下采样 1280->128
    print("coeffs:"+str(len(coeffs)))
    for i in range(len(coeffs)):
        print("coeffs[{}]:{}".format(i,len(coeffs[i])))
    coeffs[-1] = np.zeros(len(coeffs[-1]))
    coeffs[-2] = np.zeros(len(coeffs[-2]))
    coeffs[0] = np.zeros(len(coeffs[0]))
    sig_filt = pywt.waverec(coeffs, 'db6') # 小波重构
    print("sig_filt:"+str(sig_filt.shape)) # 一维的（1280，）
    return sig_filt
~~~

#### 1.2数据分类

 根据AAMI标准划分5类，N，V，S，F，Q。

![img](https://img-blog.csdn.net/20170620142038162)  

#### 1.3数据切片方式及基本设定

采用信号切片的方式，每个片段长5s 。若一个切片中出现多种类型心拍的话，那标签应该怎么打呢？这一点就值得商榷了。这里我们采取这样的一种规则来缓解一下这个问题：

1. 一个切片中所有心拍为正常时，该切片为正常；
2. 一个切片中同时存在正常和异常心拍，该切片为异常；
3. 一个切片中同时存在多类异常心拍，以切片中最多的异常类型为该切片类型；
4. 一个切片中的存在多类且相同数目的异常心拍，以最先出现的异常类型为该切片类型。

根据以上的设定，在训练集中5类样本数分别为N：5907，V：1633，S：344，F：54，Q：4。得益于我们上述的切片方式，这里可以通过片段间的互相重叠来缓解类别不平衡所带来的影响，即样本量少的类别相邻切片间多重叠一些，能采集的样本就多一些。相比简单的过采样得到完全一致的新样本，这样的方式得到的新样本与原样本之间还能存在一定差异，理论上质量更好。以不重叠情况下最多的类别为基准，其余类别切片重叠长度可由以下公式估算：

![img](https://img-blog.csdnimg.cn/20190320202949314.png)

其中，*ol*表示要重叠的长度，*round*表示四舍五入，*L*表示每个切片的长度，*n*表示当前类别的样本数目，*N*表示数目最多类别的样本数。

对于截取好的切片，还要再进行小波变换滤波（滤掉基线）和z-score标准化（均值为0，标准差为1） ![img](http://latex.codecogs.com/gif.latex?x%5E%7B*%7D%20%3D%20%5Cfrac%7Bx-%5Cbar%7Bx%7D%7D%7B%5Csigma%20%7D) 。另外，网络中存在多次的2倍下采样，而我们的5s片段在数据库360 Hz采样率的情况下长为1800，3次下采样后便不能整除，因此这里将所有片段再次重采样为1280点，等效256 Hz采样率。这个采样率也是足够的，原文中所采用的信号采样率也只有200 Hz。 

最终数据被划分为训练集，验证机，测试集

~~~python
五个种类在第一维上进行合并
TrainX.shape: (27003, 1280, 1)
TestX.shape: (7942, 1280, 1)
TrainY.shape: (27003, 5)
TestY.shape: (7942, 5)
TrainX的80%用来做训练，20%用来做验证，TestX全集是测试集合
~~~

### 2.模型搭建

基于keras，搭建了一个Resnet18的一维卷积网络，结构如下：

![](C:\Users\13018\Desktop\专利底稿\Resnet18.jpg)

### 3.模型预测

