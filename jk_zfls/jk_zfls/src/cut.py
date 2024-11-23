
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os
import shutil

# 定义鼠标回调函数，获取鼠标在图片上的坐标
def show_mouse_position_clien(image):
    def show_mouse_position(event, x, y, flags, param):
        if event == cv2.EVENT_MOUSEMOVE:
            # 获取像素值
            pixel_value = image[y, x]
            # 显示坐标和像素值
            print(f"X: {x}, Y: {y}, Pixel Value: {pixel_value}")

    cv2.imshow("Image", image)
    cv2.setMouseCallback("Image", show_mouse_position)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def cut_image(image, preview=False):
    """
    Cuts a given image into 50x50 patches.

    Parameters:
    - image: PIL.Image.Image, numpy array, or file path.
    - preview: bool, whether to preview a few patches.

    Returns:
    - patches: list of PIL.Image.Image, 50x50 patches of the image.
    """
    # 检查输入是否为文件路径
    if isinstance(image, str):
        image = Image.open(image)

    # 如果是 numpy 数组，转换为 PIL.Image
    if isinstance(image, np.ndarray):
        # 确保数据类型为 uint8
        if image.dtype != np.uint8:
            image = image.astype(np.uint8)

        # 检查是否是三通道 RGB 图像
        if len(image.shape) == 2:  # 灰度图
            image = np.stack([image] * 3, axis=-1)
        elif len(image.shape) == 3 and image.shape[2] != 3:
            raise ValueError("输入的 numpy 数组需要是 RGB 图像（H, W, 3）")

        image = Image.fromarray(image)

    # 检查是否为 PIL.Image.Image 类型
    if not isinstance(image, Image.Image):
        raise TypeError("输入必须是 PIL.Image, numpy 数组, 或文件路径")

    # Dimensions of each patch
    patch_size = 50
    img_width, img_height = image.size

    # Calculate the number of patches in each dimension
    num_patches_x = img_width // patch_size
    num_patches_y = img_height // patch_size

    patches = []
    for i in range(num_patches_y):
        for j in range(num_patches_x):
            left = j * patch_size
            upper = i * patch_size
            right = left + patch_size
            lower = upper + patch_size
            patch = image.crop((left, upper, right, lower))
            patches.append(patch)

    # Preview a few patches
    if preview:
        cols = 5
        rows = min(len(patches) // cols, 2)
        fig, axes = plt.subplots(rows, cols, figsize=(15, 6))
        for idx, ax in enumerate(axes.flatten()[:rows * cols]):
            ax.imshow(patches[idx])
            ax.axis("off")
        plt.show()

    return patches


if __name__ == '__main__':
    image = cv2.imread('../competition_plot1.png')
    print(image.shape)
    # show_mouse_position_clien(image)
    # startx, starty, endx, endy = 145, 59, 513, 428
    # cut_image(image, startx, starty, endx, endy)
