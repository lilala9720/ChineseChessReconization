# 说明:
## 文件内容
data_base: 存放用于匹配的基础图片

Images_1: 最开始好好摆放的图片

images_2: 各种倾斜，光照，模糊图片

images_1_out: Images_1 sift识别结果图 

images_2_out: images_2 sift识别结果图

images_1_out_surf: Images_1 surf识别结果图 

images_2_out_surf: images_2 surf识别结果图

识别结果图中的白色圆圈为检测出的棋盘中的圆
## 运行测试:
demo.py文件15行，分别使用sift 和 surf 提取特征测试

demo.py文件309行，选择 测试图片所在的目录 和 输出结果图片输出目录，然后执行即可在输出目录查看结果


## 测试新的图片: 
图片必须是在棋盘四个角打上蓝色点 将图片放入draw_images目录下，执行draw_image.py

点击鼠标左键，在如上所示的四个点的位置点击一下鼠标左键。点击完四个点后，按回车键确认即可。原始图片就被打上点了，如果点错了，可以按 “q” 键取消。