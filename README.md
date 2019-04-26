# -
第一个github项目，还不太会用。

网络很简单，三层卷积+两层全连接。

有四个任务可以训练：
1.识别是否带墨镜。
2.识别人脸朝向（直视、左、右、上）。
3.识别表情（自然、生气、高兴、伤心）。
4.识别人脸（20个人）。

表情识别很难做，因为数据集规模太小，且除了“高兴”之外，其他表情特征非常不明显。
其他三个任务都不难，识别率都可以到90%以上，人脸识别可以到97%以上。


数据集介绍：
cmu的一次课程用的数据集，大概几百张图片，数据集很小。
位置：http://www.cs.cmu.edu/afs/cs.cmu.edu/user/mitchell/ftp/faces.html   trainset_directory
已经下载好了，删掉一些文件，放到faces.zip中了，推荐解压后直接使用，到链接中下载也行，不过要保证解压后的faces文件夹里只有20个文件夹，没有其他文件。

需要额外创建一些文件夹: ckpt_direction  ckpt_emotion  ckpt_person  ckpt_sunglasses  test  train



