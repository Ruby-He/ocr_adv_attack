import os
import torch
import shutil
import utils
from crnn import CRNN
from torch.autograd import Variable
from torchvision import utils as vutils
from dataset import test_dataset_builder
import time
import sys
from utils import Logger

# Hyperparameter
eps = 0.13
alpha = 1
iters = 3
Height = 32
Width = 100
batch_size = 1
use_cuda=True
model_path = './checkpoints/crnn.pth'     # 英文模型权重路径
out = './PGD-result'
test_img_path = '/data/hyr/dataset/advGAN-data/100'
Accuracy = './PGD-result/accuracy.txt'
output_path = './PGD-result/perdict.txt'

# alphabet = '0123456789abcdefghijklmnopqrstuvwxyz'
alphabet = '0123456789abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ#*?'

if not os.path.exists(out):
    os.makedirs(out)
# del all the output directories and files
del_list = os.listdir(out)
for f in del_list:
    file_path = os.path.join(out, f)
    if os.path.isfile(out):
        os.remove(out)
    elif os.path.isdir(out):
        shutil.rmtree(out)

if not os.path.exists('./PGD-result/output-img/org'):
    os.makedirs('./PGD-result/output-img/org')
if not os.path.exists('./PGD-result/output-img/adv'):
    os.makedirs('./PGD-result/output-img/adv')
if not os.path.exists('./PGD-result/attack-suc-img'):
    os.makedirs('./PGD-result/attack-suc-img')

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# create model
model = CRNN(32, 3, len(alphabet) + 1, 256).to(device)  
model.eval() 

# Load pre-train model
model.load_state_dict(torch.load(model_path))

# 创建转换器，测试阶段用于将ctc生成的路径转换成最终序列
converter = utils.strLabelConverter(alphabet, ignore_case=False)

# ctcloss
ctc_loss = torch.nn.CTCLoss(blank=0, reduction='mean', zero_infinity=False)

# test_dataset
test_dataset = test_dataset_builder(Height, Width, test_img_path)
test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=4)

def pgd(model, device, test_dataloader, eps, alpha, iters):
    result = dict()
    test_len = len(test_dataloader)
    k = 0
    for i, data in enumerate(test_dataloader):
        images = data[0]
        labels = data[1]
        img_index_ori = data[2][0]
        img_index_adv = data[3][0]
        images = images.to(device)
        # images.requires_grad = True
        
        # save ori_image
        vutils.save_image(images, "./PGD-result/output-img/org/org_{}_{}.png".format(labels[0],img_index_ori), normalize=True)
        for j in range(iters):
            images = images.clone().detach().requires_grad_(True)
            # ctcloss的参数(preds, target, preds_size, length)
            torch.backends.cudnn.enabled=False     
            preds = model(images)                          # crnn识别
            preds_len = torch.IntTensor([preds.size(0)] * preds.size(1))
            labels = [str(i).lower() for i in labels]
            targets, target_len = converter.encode(labels)
            # Calculate the loss
            loss = ctc_loss(preds, targets, preds_len, target_len)

            # Zero all existing gradients
            model.zero_grad()
        
            # Calculate gradients of model in backward pass
            loss.backward()

            # Collect datagrad
            data_grad = images.grad.data
            
            # Collect the element-wise sign of the data gradient 
            sign_data_grad = data_grad.sign()

            pertubation = torch.clamp(eps*sign_data_grad, min=-alpha, max=alpha)   
            images = torch.clamp(images + pertubation, min=0, max=1)
            # images = adv_image
            # return images
            # print(images.size())
        # save perturbed_image 
        adv_image = images        
        vutils.save_image(adv_image, 
            "./PGD-result/output-img/adv/{}_{}_adv.png".format(img_index_ori,labels[0]), normalize=True)

        output = model(adv_image)
        _, output = output.max(2)           # 取可能性最大的indecis size
        output = output.transpose(1, 0).contiguous().view(-1)  # 转成以为索引列表
        output_size = torch.IntTensor([output.size(0)])        # 转成字符序列
        sim_output = converter.decode(output.data, output_size.data, raw=False)
        if labels[0] != sim_output:
            k = k + 1
            result[img_index_ori] = "{}: {} -> {}\n".format(img_index_adv, labels[0], sim_output)
            vutils.save_image(adv_image, 
            "./PGD-result/attack-suc-img/{}_{}_adv.png".format(img_index_ori,labels[0]), normalize=True)
    result = sorted(result.items(), key=lambda x:x[0])

    # 按规则保存攻击后识别结果        
    with open(output_path, 'w') as f:
        for item in result:
            f.write(item[1])
    acc = str(k / test_len)
    print(acc)
    with open(Accuracy, "w") as f:           
        f.write(output_path + ' ' + 'accuracy:' + acc + '\n')
    
if __name__ == '__main__':
    time_start = time.time()
    pgd(model, device, test_dataloader, eps, alpha, iters)
    time_end = time.time()
    time_sum = time_end - time_start
    print('144 sample generation time:' + str(time_sum))