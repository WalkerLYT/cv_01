import gradio as gr
import exifread
import os
from datetime import datetime

def save_image(image):
    # 定义保存图像的目录和文件名
    save_dir = "img"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(save_dir, f"{timestamp}.jpg")

    # 如果存在EXIF，则保存图像并保留 EXIF 信息
    if image.info.get('exif'):
        exif_data = image.info['exif']
        image.save(save_path, exif=exif_data)
    else:
        image.save(save_path)
    return save_path

def dms_to_decimal(degrees, minutes, seconds, direction):
    decimal = degrees + minutes / 60 + seconds / 3600
    if direction in ['S', 'W']:
        decimal = -decimal
    return round(decimal, 4)


def analyze_image(image):
    # 先将图片保存到本地（不然Image类型无法直接转换为数据流给exifread去读取）
    img_path = save_image(image)
    with open(img_path, 'rb') as image_file:
        # 读取 EXIF 数据
        exif_data = exifread.process_file(image_file)
        data_keys = list(exif_data.keys())
    if 'GPS GPSLatitude' in data_keys and 'GPS GPSLongitude' in data_keys:  # 判断目标数据是否存在，这里需要的是经纬度数据
        GPSLatitude = exif_data['GPS GPSLatitude']
        GPSLongitude = exif_data['GPS GPSLongitude']
        GPSLatitudeRef = exif_data['GPS GPSLatitudeRef'].values
        GPSLongitudeRef = exif_data['GPS GPSLongitudeRef'].values
        # 转化为数字
        gps_lat = GPSLatitude.values
        gps_long = GPSLongitude.values
        lat = dms_to_decimal(float(gps_lat[0].num) / float(gps_lat[0].den),
                             float(gps_lat[1].num) / float(gps_lat[1].den),
                             float(gps_lat[2].num) / float(gps_lat[2].den),
                             GPSLatitudeRef)
        long = dms_to_decimal(float(gps_long[0].num) / float(gps_long[0].den),
                              float(gps_long[1].num) / float(gps_long[1].den),
                              float(gps_long[2].num) / float(gps_long[2].den),
                              GPSLongitudeRef)

        res = f"拍摄地点：\n纬度：{lat}° {GPSLatitudeRef}\n经度：{long}° {GPSLongitudeRef}"
        return res
    else:
        return "图片不存在位置信息"


with gr.Blocks() as demo:
    with gr.Tab("img2Text"):
        with gr.Row():
            with gr.Column():
                input_img = gr.Image(sources=["upload"], label="上传图片", type="pil")
                btn1 = gr.Button("解析")
            output_label = gr.Textbox(label="图像信息")
        btn1.click(analyze_image, inputs=input_img, outputs=output_label)

# 启动 Gradio 应用，创建可共享链接
demo.launch(share=True)
