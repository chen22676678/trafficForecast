from PIL import Image, ImageOps

# 打开PNG图像
img = Image.open('整体流程.png')

# 转换为黑白色
img = img.convert('L')
img.save('整体流程1.png')
# 黑白反转
# img_inverted = ImageOps.invert(img)
# 保存为新的PNG文件

# img_inverted.save('demo.png')
