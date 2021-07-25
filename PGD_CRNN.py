import torch
from dataset import *
from utils import *
from PIL import Image
import models.crnn as crnn
import numpy as np 
import torch.optim as optim
from  torchvision import utils as vutils
# from Chinese_alphabet import alphabet
use_cuda=True
model_path = './data/crnn.pth'  # 英文模型权重路径
# model_path = './data/netCRNN.pth'  # 中文模型权重路径
img_path = './data/demo.png'  # 测试图片路径
alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

model = crnn.CRNN(32, 1, len(alphabet) + 1, 256).to(device)  # 创建模型

# 加载预训练模型
print('loading pretrained model from %s' % model_path)
model.load_state_dict(torch.load(model_path))

# 创建转换器，测试阶段用于将ctc生成的路径转换成最终序列，使用英文字典时忽略大小写
converter = strLabelConverter(alphabet, ignore_case=False)

# 图像大小转换器
transformer = resizeNormalize((100, 32))

# 读取并转换图像大小为100 x 32 w x h
image = Image.open(img_path).convert('L')#参数'L'表示将图像转为灰度图
image = transformer(image) # (1, 32, 100)
image = image.view(1, *image.size())  # (b, c, h, w) (1, 1, 32, 100)
model.eval()

# CTCLoss
criterion = torch.nn.CTCLoss()

# PGD攻击，eps是限制添加扰动大小，alpha是添加扰动的参数，iters是PGD循环轮次
def pgd_attack(model, image, eps=0.3, alpha=0.2, iters=1):
    
    data = image
    data = data.to(device)

    for i in range(iters):    
        preds = model(data)
        preds = Variable(preds, requires_grad=True)
        # ctcloss的参数
        batch_size = data.size(0)
        target, length = converter.encode(alphabet)  # encode函数，提供target和target的length
        preds_size = torch.IntTensor([preds.size(0)] * batch_size)
        # 模型梯度清零
        model.zero_grad()
        # Calculate the loss       
        loss = criterion(preds, target, preds_size, length)        
        # Calculate gradients of model in backward pass        
        loss.backward()
        # Collect datagrad
        data_grad = preds.grad
        # Collect the element-wise sign of the data gradient
        sign_data_grad = data_grad.sign()            
        sign_data_grad = sign_data_grad.resize_([1,32,100])
        # 生成对抗样本
        adv_image = data + alpha*sign_data_grad
        eta = torch.clamp(adv_image - data, min=-eps, max=eps)      
        data = torch.clamp(data + eta, min=0, max=1).detach_()
        perturbed_image = data
        # save perturbed_image
        vutils.save_image(perturbed_image, './data/adv_pgd.png', normalize=True)
        print("eps：", alpha, "nb_iter：", i)
        print_function(perturbed_image)
           
    return perturbed_image

# 输出函数
def print_function(perturbed_image):
    output = model(perturbed_image)
    _, output = output.max(2)  # 取可能性最大的indecis size
    output = output.transpose(1, 0).contiguous().view(-1)  # 转成以为索引列表
    # 转成字符序列
    output_size = torch.IntTensor([output.size(0)])
    raw_output = converter.decode(output.data, output_size.data, raw=True)
    sim_output = converter.decode(output.data, output_size.data, raw=False)
    print('%-20s => %-20s' % (raw_output, sim_output))

if __name__ == '__main__':
    pgd_attack(model, image)