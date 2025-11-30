from PIL import Image, ImageDraw

img = Image.open("test_image/screenshot.png")
draw = ImageDraw.Draw(img)
# 在 (143, 748) 画一个红色圆圈
draw.ellipse((138, 743, 148, 753), fill="red")
img.save("test_image/marked.png")