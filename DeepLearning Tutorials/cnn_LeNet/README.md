CNN卷积神经网络代码详细解读（基于python+theano）
===
这个代码实现的是一个简化了的LeNet5，具体如下：  

- 没有实现location-specific gain and bias parameters  
- 用的是maxpooling，而不是average_pooling  
- 分类器用的是softmax，LeNet5用的是rbf  
- LeNet5第二层并不是全连接的，本程序实现的是全连接  

另外，代码里将卷积层和子采用层合在一起，定义为“LeNetConvPoolLayer“（卷积采样层），这好理解，因为它们总是成对出现。但是有个地方需要注意，代码中将卷积后的输出直接作为子采样层的输入，而没有加偏置b再通过sigmoid函数进行映射，即没有了下图中fx后面的bx以及sigmoid映射，也即直接由fx得到Cx。  

最后，代码中第一个卷积层用的卷积核有20个，第二个卷积层用50个，而不是上面那张LeNet5图中所示的6个和16个。  

了解了这些，下面看代码：  

（1）导入必要的模块
---
```python
import cPickle  
import gzip  
import os  
import sys  
import time  
  
import numpy  
  
import theano  
import theano.tensor as T  
from theano.tensor.signal import downsample  
from theano.tensor.nnet import conv 
```
（2）定义CNN的基本"构件"
---
CNN的基本构件包括卷积采样层、隐含层、分类器，如下  
###<center>定义LeNetConvPoolLayer（卷积+采样层）</center>  
见代码注释：  
```python
""" 
卷积+下采样合成一个层LeNetConvPoolLayer 

    :type rng: numpy.random.RandomState
    :param rng: 随机数，用于初始化weights

    :type input: theano.tensor.dtensor4
    :param input: 4维向量，表示图像大小

    :type filter_shape: tuple or list of length 4
    :param filter_shape: (number of filters, num input feature maps,
                          filter height, filter width)

    :type image_shape: tuple or list of length 4
    :param image_shape: (batch size, num input feature maps,
                         image height, image width)

    :type poolsize: tuple or list of length 2
    :param poolsize: the downsampling (pooling) factor (#rows, #cols)
"""  
class LeNetConvPoolLayer(object):  
    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):  
    
#assert condition，condition为True，则继续往下执行，condition为False，中断程序  
#image_shape[1]和filter_shape[1]都是num input feature maps，它们必须是一样的。  
        assert image_shape[1] == filter_shape[1]  
        self.input = input  
  
#每个隐层神经元（即像素）与上一层的连接数为num input feature maps * filter height * filter width。  
#可以用numpy.prod(filter_shape[1:])来求得  
        fan_in = numpy.prod(filter_shape[1:])  
  
#lower layer上每个神经元获得的梯度来自于：num output feature maps * filter height * filter width /pooling size  
        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /  
                   numpy.prod(poolsize))  
                     
#以上求得fan_in、fan_out ，将它们代入公式，以此来随机初始化weights(线性卷积核)
        W_bound = numpy.sqrt(6. / (fan_in + fan_out))  
        self.W = theano.shared(  
            numpy.asarray(  
                rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),  
                dtype=theano.config.floatX  
            ),  
            borrow=True  
        )  

#偏置b是一维向量，每个输出图的特征图都对应一个偏置，  
#而输出的特征图的个数由filter个数决定，因此用filter_shape[0]即number of filters来初始化  
        b_values = numpy.zeros((filter_shape[0],), dtype=theano.config.floatX)  
        self.b = theano.shared(value=b_values, borrow=True)  
  
#将输入图像与filter卷积，conv.conv2d函数  
#卷积完没有加b再通过sigmoid，这里是一处简化。  
        conv_out = conv.conv2d(  
            input=input,  
            filters=self.W,  
            filter_shape=filter_shape,  
            image_shape=image_shape  
        )  
  
#maxpooling，最大子采样过程  
        pooled_out = downsample.max_pool_2d(  
            input=conv_out,  
            ds=poolsize,  
            ignore_border=True  
        )  
  
#加偏置，再通过tanh映射，得到卷积+子采样层的最终输出  
#因为b是一维向量，这里用维度转换函数dimshuffle将其reshape为(1,10,1,1)。
#每个偏置将会在整个最小批次的特征图像上进行处理
        self.output = T.tanh(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))  
#卷积+采样层的参数  
        self.params = [self.W, self.b]  
#保持追踪
        self.input=input
```

###<center>定义隐含层HiddenLayer</center>
 
隐藏层的输入即input，输出即隐藏层的神经元个数。输入层与隐藏层是全连接的。  
假设输入是$n_{in}$维的向量（也可以说时$n_{in}$个神经元），隐藏层有$n_{out}$个神经元，则因为是全连接， 
一共有$n_{in}*n_{out}$个权重，故$W$大小是($n_{in}$,$n_{out}$),$n_{in}$行$n_{out}$列，每一列对应隐藏层的每一个神经元的连接权重。  
b：偏置，隐藏层有$n_{out}$个神经元，故$b$时$n_{out}$维向量。  
rng：随机数生成器，numpy.random.RandomState，用于初始化$W$。  
input：训练模型所用到的所有输入，并不是MLP的输入层，MLP的输入层的神经元个数时n_in，而这里的参数input大小是（$n_{example}$,$n_{in}$），每一行一个样本，即每一行作为MLP的输入层。  
activation：激活函数，这里定义为函数tanh。  

代码要兼容GPU，则必须使用 dtype=theano.config.floatX,并且定义为theano.shared 
另外，W的初始化有个规则：如果使用tanh函数，则在$-\sqrt{6.\over (n_{in}+n_{hidden})}$到$\sqrt{6.\over (n_{in}+n_{hidden})}$之间均匀。  
抽取数值来初始化W，若时sigmoid函数，则以上再乘4倍。  
```python
class HiddenLayer(object):  
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,  
                 activation=T.tanh):  
#类HiddenLayer的input即所传递进来的input  
         self.input = input
  
#如果W未初始化，则根据上述方法初始化。  
#加入这个判断的原因是：有时候我们可以用训练好的参数来初始化W。  
         if W is None:  
            W_values = numpy.asarray(  
                rng.uniform(  
                    low=-numpy.sqrt(6. / (n_in + n_out)),  
                    high=numpy.sqrt(6. / (n_in + n_out)),  
                    size=(n_in, n_out)  
                ),  
                dtype=theano.config.floatX  
            )  
            if activation == theano.tensor.nnet.sigmoid:  
                W_values *= 4  
            W = theano.shared(value=W_values, name='W', borrow=True)  
  
         if b is None:  
            b_values = numpy.zeros((n_out,), dtype=theano.config.floatX)  
            b = theano.shared(value=b_values, name='b', borrow=True)  
  
#用上面定义的W、b来初始化类HiddenLayer的W、b  
         self.W = W  
         self.b = b  

#隐含层的输出  
         lin_output = T.dot(input, self.W) + self.b  
         self.output = (  
            lin_output if activation is None  
            else activation(lin_output)  
         )  
  
#隐含层的参数  
         self.params = [self.W, self.b]  
```

###<center>定义分类器 （Softmax回归）</center>

定义分类层LogisticRegression，也即Softmax回归。   
在deeplearning tutorial中，直接将LogisticRegression视为Softmax，   
而我们所认识的二类别的逻辑回归就是当$n_{out}=2$时的LogisticRegression。   


```python
#参数说明：  
#input：大小就是(n_example,n_in)，其中n_example是一个batch的大小（因为我们训练时用的是Minibatch SGD，因此input这样定义）  
#n_in,即上一层(隐含层)的输出  
#n_out,输出的类别数   
class LogisticRegression(object):  
    def __init__(self, input, n_in, n_out):  
  
#W大小是n_in行n_out列，b为n_out维向量。即：每个输出对应W的一列以及b的一个元素。    
        self.W = theano.shared(  
            value=numpy.zeros(  
                (n_in, n_out),  
                dtype=theano.config.floatX  
            ),  
            name='W',  
            borrow=True  
        )  
  
        self.b = theano.shared(  
            value=numpy.zeros(  
                (n_out,),  
                dtype=theano.config.floatX  
            ),  
            name='b',  
            borrow=True  
        )  
  
#input是(n_example,n_in)，W是（n_in,n_out）,点乘得到(n_example,n_out)，加上偏置b，  
#再作为T.nnet.softmax的输入，得到p_y_given_x  
#故p_y_given_x每一行代表每一个样本被估计为各类别的概率      
#PS：b是n_out维向量，与(n_example,n_out)矩阵相加，内部其实是先复制n_example个b，  
#然后(n_example,n_out)矩阵的每一行都加b  
        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)  
  
#argmax返回最大值下标，因为本例数据集是MNIST，下标刚好就是类别。axis=1表示按行操作。  
        self.y_pred = T.argmax(self.p_y_given_x, axis=1)  
  
#params，LogisticRegression的参数       
        self.params = [self.W, self.b]  
```
到这里，CNN的基本”构件“都有了，下面要用这些”构件“组装成LeNet5（当然，是简化的，上面已经说了），具体来说，就是组装成：LeNet5=input+LeNetConvPoolLayer_1+LeNetConvPoolLayer_2+HiddenLayer+LogisticRegression+output。  
然后将其应用于MNIST数据集，用BP算法去解这个模型，得到最优的参数。  
###<center>（3）加载MNIST数据集</center>
```python
def load_data(dataset):  
    # dataset是数据集的路径，程序首先检测该路径下有没有MNIST数据集，没有的话就下载MNIST数据集  
    # 这一部分就不解释了，与softmax回归算法无关。  
    data_dir, data_file = os.path.split(dataset)  
    if data_dir == "" and not os.path.isfile(dataset):  
        # Check if dataset is in the data directory.  
        new_path = os.path.join(  
            os.path.split(__file__)[0],  
            "..",  
            "data",  
            dataset  
        )  
        if os.path.isfile(new_path) or data_file == 'mnist.pkl.gz':  
            dataset = new_path  
  
    if (not os.path.isfile(dataset)) and data_file == 'mnist.pkl.gz':  
        import urllib  
        origin = (  
            'http://www.iro.umontreal.ca/~lisa/deep/data/mnist/mnist.pkl.gz'  
        )  
        print 'Downloading data from %s' % origin  
        urllib.urlretrieve(origin, dataset)  
  
    print '... loading data'  
#以上是检测并下载数据集mnist.pkl.gz，不是本文重点。下面才是load_data的开始  
      
#从"mnist.pkl.gz"里加载train_set, valid_set, test_set，它们都是包括label的  
#主要用到python里的gzip.open()函数,以及 cPickle.load()。  
#‘rb’表示以二进制可读的方式打开文件  
    f = gzip.open(dataset, 'rb')  
    train_set, valid_set, test_set = cPickle.load(f)  
    f.close()  
     
  
#将数据设置成shared variables，主要时为了GPU加速，只有shared variables才能存到GPU memory中  
#GPU里数据类型只能是float。而data_y是类别，所以最后又转换为int返回  
    def shared_dataset(data_xy, borrow=True):  
        data_x, data_y = data_xy  
        shared_x = theano.shared(numpy.asarray(data_x,  
                                               dtype=theano.config.floatX),  
                                 borrow=borrow)  
        shared_y = theano.shared(numpy.asarray(data_y,  
                                               dtype=theano.config.floatX),  
                                 borrow=borrow)  
        return shared_x, T.cast(shared_y, 'int32')  
  
  
    test_set_x, test_set_y = shared_dataset(test_set)  
    valid_set_x, valid_set_y = shared_dataset(valid_set)  
    train_set_x, train_set_y = shared_dataset(train_set)  
  
    rval = [(train_set_x, train_set_y), (valid_set_x, valid_set_y),  
            (test_set_x, test_set_y)]  
    return rval  
```