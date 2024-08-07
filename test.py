import os
import torch
from torch import nn
from PIL import Image
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import open_clip

# 设置环境变量以禁用符号链接警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# 加载 CLIP 模型
clip_model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')

# 加载 GPT-2 模型和分词器
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
gpt2_model = GPT2LMHeadModel.from_pretrained('gpt2')

# 将模型设置为评估模式
clip_model.eval()
gpt2_model.eval()

# 创建线性变换层将 512 维的图像特征映射到 768 维
linear_transform = nn.Linear(512, 768)


def extract_image_features(image_path):
    # 打开图像并进行预处理
    image = Image.open(image_path).convert("RGB")
    image = preprocess(image).unsqueeze(0)  # 预处理并增加批量维度

    # 提取图像特征
    with torch.no_grad():
        image_features = clip_model.encode_image(image)

    return image_features


def generate_caption(image_features):
    # 将图像特征转换为适合 GPT-2 模型的输入
    image_features = image_features.squeeze(0)  # 去除批量维度

    # 使用线性变换将图像特征映射到 768 维
    image_features = linear_transform(image_features)

    # 将图像特征的形状从 [512] 调整为 [1, 1, 768]
    image_features = image_features.unsqueeze(0).unsqueeze(0)

    # 创建初始的文本提示
    text_inputs = tokenizer("Image: ", return_tensors="pt")

    # 获取 GPT-2 模型的输入嵌入
    inputs_embeds = gpt2_model.transformer.wte(text_inputs['input_ids'])

    # 将图像特征嵌入拼接到文本提示后面
    inputs_embeds = torch.cat([inputs_embeds, image_features], dim=1)

    # 生成描述
    output = gpt2_model.generate(
        input_ids=None,
        inputs_embeds=inputs_embeds,
        max_length=50,
        num_return_sequences=1,
        no_repeat_ngram_size=2,
        top_k=50,
        top_p=0.95,
        temperature=1.0
    )

    # 解码生成的描述
    caption = tokenizer.decode(output[0], skip_special_tokens=True)
    return caption


image_path = "img/20240727_004312.jpg"
image_features = extract_image_features(image_path)
caption = generate_caption(image_features)
print("Generated Caption:", caption)
