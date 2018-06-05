

### 5.1.1 为什么选择序列模型 

 `$\quad$`序列模型的应用：
 - 语音识别：将输入的语音信号直接转化为文本信息，语音信号和文本信号都是序列数据 
 - 音乐生成：生成音乐乐谱，输出的乐谱是序列数据，输入可以是空格或者整数 
 - 情感分类：将数据的句子转化为相应的等级或评分，输入是一个序列
 - DNA序列分析 ：找到输入DNA序列的蛋白质子序列
 - 机器翻译：两种不同语言之间的相互转换
 - 视屏行为识别：
 - 命名实体识别：从输入的句子中识别实体的名字 
 
![image](https://img-blog.csdn.net/20180209155912605?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvS29hbGFfVHJlZQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70)


### 5.1.2 数学符号 

- 输入x : `$x^{<t>}$`表示输入x中的第 t 个符号 
- 输出y :  `$y^{<t>}$`表示输出y中的第 t 个符号 
- `$T_x$`表示输入x的长度 
- `$T_y$`表示输入y的长度 
- `$x^{(i)<t>}$` 表示第i个样本的第t个符号  
- 利用单词字典编码来表示每一个输入符号，如one-hot编码等，实现输入与输出之间的映射关系 



### 5.1.3 循环神经网络 

`$\quad$`为什么不使用标准的神经网络？ 

**Problems:**
- 在不同的样本中输入和输出的长度可能并不相同 
- 通过不同位置文本学习到的特征不能共享 

![image](https://img-blog.csdn.net/20180303160705321?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvS29hbGFfVHJlZQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



`$\quad$` RNN在处理序列数据时不存在上面的两个问题，在每一个时间步中，会传递一个激活值到下一个时间步用于下一个时间步的计算，如下图所示： 


![image](https://note.youdao.com/yws/public/resource/0a6a3919b29a277948adec0a6c3632b1/xmlnote/C9AD9493D4674308A821E7E180FC370C/5626)

在 0 时刻，需要构造一个激活值，通常是一个零向量，循环神经网络从左到右扫描数据，同时 ==**共享每个时间步的参数**==

- `$W_{aa}$` 管理激活层到隐含层的连接 
- `$W_{ax}$` 管理输入到隐含层的连接
- `$W_{ya}$` 管理隐含层到激活值 `$y_{<t>}$`的连接 

每个输出只使用了前面的输入信息没有使用后向的 ,BRNN可以解决这个问题 

##### RNN的前向传播

- 构造初始激活向量： `$a^{<0>}=\vec 0$`
- 计算 `$a^{<1>}=g(W_{aa}a^{<0>}+W_{ax}x^{<t>}+b_a)$` ,激活函数通常选tanh ，也可以选择relu
- 计算激活值 `$\hat y^{<t>}=g(W_{ya}a^{<t>}+b_y)$` ,如果是多分类，激活函数选择softmax  


对上述公式进行简化：

```math
a^{<t>} = g(W_a [a^{<t-1>},x^{<t>}]+b_y) 

\hat y^{<t>} = g(W_ya^{<t>}+b_y)
```

其中：

```math
W_a=[W_{aa}:W_{ax}]
```


```math
[a,x]=\left[\begin{matrix}a^{<t-1>} \\x^{<t>}\end{matrix}\right]
```



![image](https://img-blog.csdn.net/20180305210613275?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvS29hbGFfVHJlZQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)

### 5.1.4 通过时间的反向传播 

`$\quad$`为了进行反向传播，使用梯度下降等方法来更新Rnn的参数，定义一个损失函数： 


```math
L^{<t>}(\hat y^{<t>},y^{<t>}) = -y^{<t>} log\hat y^{<t>} -(1-y^{<t>})log(1-\hat y^{<t>})

L(y,\hat y) = \sum_{t=1}^{T_y} L^{<t>}(\hat y^{<t>},y^{<t>})
```

将每个输出的损失进行求和，通过对其进行导数计算来更新参数。 



### 5.1.5 不同类型的循环神经网络 

`$\quad$` 对于RNN ，不同的问题需要不同的输入输出结构 
- [x] `$many -to -many(T_x=T_y)$` ，输入和输出的长度相同： 



- [x]  `$many-to-one$`  ,例如想情感分类中，我们最终只需要得出这个句子的正负判别或打星操作，所以只要最终的一个输出就可以了： 


- [x] `$one-to-many$`,例如在音乐生成中，我们输入一个音乐的类型或者空值，输出一段音符序列


- [x] `$many -to -many(T_x<>T_y)$` ,像机器翻译这种，输入和输出的长度可能不同，这是一种多对多的结构： 

![image](https://note.youdao.com/yws/public/resource/0a6a3919b29a277948adec0a6c3632b1/xmlnote/622083AFD5EA4A04AFC56CF242470ADE/5802)





### 5.1.6 语言模型和序列生成 
`$\quad$`什么是语言模型？ 比如下面这个例子，两个单词相同的发音，如何让构建的语音识别系统能够正确的识别输出，需要模型评估出输入句子各种单词出现的可能性大小

![image](https://note.youdao.com/yws/public/resource/0a6a3919b29a277948adec0a6c3632b1/xmlnote/C6B4C1E1B1954BB29A909FCA48C243E0/5813)

- [x] 使用RNN 构建 语言模型： 
    - 训练集：一个很大的语言文本语料库
    - Tokenize： 将句子使用字典库标记化，其中，未出现在字典库中的词使用"UNK"来表示 
    - 使用零向量对输出进行预测，预测第一个出现的单词的可能性
    - 通过前面的输出作为输入，逐步预测后面单词出现的概率 
    - 训练网络，计算损失函数，更新网络参数
    


### 5.1.7 对序列采样  

`$\quad$` 在对模型训练好后，需要了解到这个模型学到了什么，一种非正式的方法是进行一次新序列采样(sample novel sequences)

对于一个训练好的RNN模型，采样步骤： 
- 首先输入 `$a^{<1>}=\vec 0,x^{<1>}=\vec0$`，经过softmax层后得到所有可能出现单词的概率分布，采用随机采获得第一个输出的单词 `$\hat y^{<1>}$`
- 进行下一个时间步，以 `$\hat y^{<1>}$` 作为第二个时间步的输入
- 当输出字典的结束标志"EOS" 或者达到设定的时间步停止 


![image](https://img-blog.csdn.net/20180303222011131?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvS29hbGFfVHJlZQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



### 5.1.8 带有神经网络的梯度消失

`$\quad$`和基本深层神经网络相似，输出y得到的梯度很难通过反向传播传播回去，很难对前面几层的权重产生影响，RNN同样存在这种问题，像下图所示，很难记住cat是单数还是复数  


![image](https://note.youdao.com/yws/public/resource/0a6a3919b29a277948adec0a6c3632b1/xmlnote/F9424D93AD5E4CF394D5276B1F5FA168/5911)

==梯度消失==很难解决，是我们在RNN中关心的问题。

`$\quad$`对于==梯度爆炸==，因为参数会指数级的增长，会使得我们的网络参数变的很大，造成数值溢出，对于梯度爆炸，解决办法就是用==梯度修剪==



### 5.1.9  GRU

#### 简化的GRU ： 

`$\quad$`引入了记忆细胞 c ,提供了长期的记忆能力：

- `$c^{<t>}=a^{<t>}$`,实际上记忆细胞输出的值就是t时间步上的激活值  
- 候选值：`$\hat c^{<t>}=tanh(W_c[c^{<t-1>},x^{<t>}]+b_c)$`
- 更新门： `$\Gamma_u=\sigma(W_u[c^{<t-1>},x^{<t>}]+b_u)$`
- 记忆细胞的更新规则：`$c^{<t>}=\Gamma_u \hat c^{<t>}+(1-\Gamma_u)c^{<t-1>}$`



![image](https://img-blog.csdn.net/20180304101019839?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvS29hbGFfVHJlZQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



#### 完整版的GRU ：
`$\quad$` 完整版的GRU 还有一个门，用来决定每个时间步的候选值 

- 记忆细胞：`$c^{<t>}=a^{<t>}$`    实际上记忆细胞输出的值就是t时间步上的激活值  

- ==相关门==：  `$\Gamma_r=\sigma(W_r[c^{<t-1>},x^{<t>}]+b_r)$`

- 候选值：`$\hat c^{<t>}=tanh(W_c[\Gamma_r* c^{<t-1>},x^{<t>}]+b_c)$`
- 更新门： `$\Gamma_u=\sigma(W_u[c^{<t-1>},x^{<t>}]+b_u)$`
- 记忆细胞的更新规则：`$c^{<t>}=\Gamma_u \hat c^{<t>}+(1-\Gamma_u)c^{<t-1>}$`


### 5.1.10 长短期记忆 LSTM 
`$\quad$`LSTM 对捕捉序列中更深层次的联系比GRU更加有效，其使用了单独的“更新门” `$\Gamma_u$`和 “遗忘门” `$\Gamma_f$`以及一个输出门  `$\Gamma_o$`,和GRU的公式对比如下图： 

![image](https://note.youdao.com/yws/public/resource/0a6a3919b29a277948adec0a6c3632b1/xmlnote/F17E76390911499AB79227C7861186B4/6020)



LSTM的可视图如下： 


- LSTM 单元 

![image](https://note.youdao.com/yws/public/resource/0a6a3919b29a277948adec0a6c3632b1/xmlnote/067EC6836CF1423CAF0C92866A996F98/5445)

- 完整的 LSTM 

![image](https://note.youdao.com/yws/public/resource/0a6a3919b29a277948adec0a6c3632b1/xmlnote/8C7D34F31F8541FA9934C8DB11B8F87C/5447)


其中，在实际的使用时，几个门值不仅仅取决于 `$a^{<t-1>},x^{<t>}$`,还可能会取决于上一个记忆细胞的值 `$c^{<t-1>}$`,这也叫偷窥孔连接


### 5.1.11 双向神经网络

`$\qquad$`双向神经网络使我们在序列的某处不仅可以获取之前的信息，还可以获取未来的信息,如下图中，根据之前的信息很难判定Teddy 是否是人名 ，在BRNN中，不仅有从左到右的前向连接层，还存在一个从右到左的反向连接层: 

![image](https://img-blog.csdn.net/20180304105617479?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvS29hbGFfVHJlZQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)



其中,预测输出值 `$\hat y^{<t>}= g(W_{y}[\vec a^{<t>}, \leftarrow a^{<t>}] + b_{y})$`,预测结果既有前向信息又有反向信息

![image](https://note.youdao.com/yws/public/resource/0a6a3919b29a277948adec0a6c3632b1/xmlnote/C5EFC953B58B4DE5B39939B6BDD1643E/6087)


### 5.1.12 深层循环神经网络 

`$\qquad$` 和深层的神经基础神经网络相似，深层RNNS也具有多层网络结构，对于一般来说三层就已经很多了，不会想传统的神经网络那样拥有很多层 ，因为RNN自身存在时间维度，结构就已经很庞大了： 

![image](https://img-blog.csdn.net/20180304112306924?watermark/2/text/aHR0cDovL2Jsb2cuY3Nkbi5uZXQvS29hbGFfVHJlZQ==/font/5a6L5L2T/fontsize/400/fill/I0JBQkFCMA==/dissolve/70/gravity/SouthEast)





参考： 

> 博客： https://blog.csdn.net/koala_tree/article/details/79299358 

> 作业： https://blog.csdn.net/koala_tree/article/details/79451127
