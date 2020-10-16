#!/usr/bin/env python3

import numpy
import glob,cv2,os.path
import caffe
from PIL import Image
# import torchvision.transforms as transform
import torch
import encoding
# import torch.nn as nn
# import torch.nn.functional as F
# from torch.autograd import Variable
# from torchvision.models import resnet

os.environ['GLOG_minloglevel'] = '0'

#### it is conflict between pytoch and caffe env ---quhongsen,2020-8-26

def get_caffe_model():

    arch = "googlenet"
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")


    root = os.path.join(father_path,"model")

    category = os.path.join(root,'categories.txt')
    prototxt = os.path.join(root,'deploy-{}.prototxt'.format(arch))
    caffemodel = os.path.join(root,'minc-{}.caffemodel'.format(arch))

    if os.path.exists(category) and os.path.join(prototxt) and os.path.exists(caffemodel):
        print("load categories and modelfile ...")
        categories = [x.strip() for x in open(category).readlines()]
        net = caffe.Classifier(prototxt, caffemodel, channel_swap=(2, 1, 0),mean=numpy.array([104, 117, 124]))  ## batch * C * H * W
    else:
        print("NotFound, check files:{},{},{}".format(category,prototxt,caffemodel))
        categories = None
        net = None

    return net, categories


# class DEPNet(nn.Module):
#     def __init__(self, nclass, backbone):
#         super(DEPNet, self).__init__()
#
#         n_codes = 8
#         self.pretrained = backbone
#         self.encode = nn.Sequential(
#             nn.BatchNorm2d(512),
#             encoding.nn.Encoding(D=512,K=n_codes),
#             encoding.nn.View(-1, 512*n_codes),
#             encoding.nn.Normalize(),
#             nn.Linear(512*n_codes, 64)
#         )
#         self.pool = nn.Sequential(
#             nn.AvgPool2d(7),
#             encoding.nn.View(-1, 512),
#             nn.Linear(512, 64),
#             nn.BatchNorm1d(64),
#         )
#         self.fc = nn.Sequential(
#             encoding.nn.Normalize(),
#             nn.Linear(64*64, 128),
#             encoding.nn.Normalize(),
#             nn.Linear(128, nclass))
#
#     def forward(self, x):
#         if isinstance(x, Variable):
#             _, _, h, w = x.size()
#         elif isinstance(x, tuple) or isinstance(x, list):
#             var_input = x
#             while not isinstance(var_input, Variable):
#                 var_input = var_input[0]
#             _, _, h, w = var_input.size()
#         else:
#             raise RuntimeError('unknown input type: ', type(x))
#
#         # pre-trained ResNet feature
#         x = self.pretrained.conv1(x)
#         x = self.pretrained.bn1(x)
#         x = self.pretrained.relu(x)
#         x = self.pretrained.maxpool(x)
#         x = self.pretrained.layer1(x)
#         x = self.pretrained.layer2(x)
#         x = self.pretrained.layer3(x)
#         x = self.pretrained.layer4(x)
#
#         # DEP head
#         x1 = self.encode(x)
#         x2 = self.pool(x)
#         x1 = x1.unsqueeze(1).expand(x1.size(0),x2.size(1),x1.size(-1))
#         x = x1*x2.unsqueeze(-1)
#         x=x.view(-1,x1.size(-1)*x2.size(1))
#         x = self.fc(x)
#
#         return x



# def get_resnet_model():
#
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#
#     current_path = os.path.abspath(__file__)
#     father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
#
#
#     root = os.path.join(father_path, "model")
#     checkpoint_path  = os.path.join(root,'resnet18_material_recognition.pth.tar')
#
#     if os.path.exists(checkpoint_path):
#         print("load pytoch modelfile ...")
#
#         classes = ['brick', 'carpet', 'ceramic', 'fabric', 'foliage', 'food', 'glass', 'hair', 'leather', 'metal',
#                'mirror', 'other', 'painted', 'paper', 'plastic', 'polishedstone', 'skin', 'sky', 'stone',
#                'tile', 'wallpaper', 'water', 'wood'] ### class-23
#
#         backbone = resnet.resnet18(pretrained=True)
#         num_class = len(classes)
#         model = DEPNet(num_class, backbone)
#
#         checkpoint = torch.load(checkpoint_path,map_location=device)
#         model.load_state_dict({k.replace('module.', ''): v for k, v in checkpoint['state_dict'].items()})
#         model.to(device)
#         model.eval()
#
#     else:
#         print("NotFound, model files:{}".format(checkpoint_path))
#         classes = None
#         model = None
#
#     return  model, classes



def material_caffe_predict(imgfile):

    model, categories = get_caffe_model()

    im = [caffe.io.load_image(imgfile) * 255.0]
    output_prob = model.predict(im)[0]

    label_index = output_prob.argmax()

    label_name = categories[label_index]
    pred_score = output_prob[label_index]

    results = {
        "scores": pred_score,### 预测分值列表，[0.2, ...]
        "pred_classes": label_name,### 预测物体编码列表，['kitchen', ...]
            }

    return results


def material_caffe_crop_predict(rgb_crop):

    model, categories = get_caffe_model()

    im = rgb_crop/255.0
    im = im.astype(numpy.float32)

    output_prob = model.predict([im*255.0])[0]
    # print(output_prob)
    label_index = output_prob.argmax()
    label_name = categories[label_index]
    pred_score = output_prob[label_index]

    results = {
        "scores": pred_score,### 预测分值列表，[0.2, ...]
        "pred_classes": label_name,### 预测物体编码列表，['kitchen', ...]
            }


    return results



# def material_pytoch_predict(imgfile):
#     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#     model, classes = get_resnet_model()
#
#     transformer = transform.Compose([transform.Resize(256),transform.CenterCrop(224),transform.ToTensor(),
#                     transform.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])])
#
#     img = Image.open(imgfile).convert('RGB')
#     img = transformer(img)
#     img = img.to(device)
#     img = img.unsqueeze(0)
#     output = model(img)
#     prob = F.softmax(output, dim=1)
#     prob = Variable(prob)
#     prob = prob.cpu().numpy()
#     label_index = numpy.argmax(prob)
#     label_name = classes[label_index]
#     pred_score = prob[0][label_index]
#
#     # print(prob)
#     pred_text = "{},{},{}".format(label_name, label_index,pred_score)
#
#     results = {
#         "scores": pred_score,### 预测分值列表，[0.2, ...]
#         "pred_classes": label_name,### 预测物体编码列表，['kitchen', ...]
#             }
#
#     return results


def test():
    test_img_path = "../demo.jpg"

    results = material_caffe_predict(test_img_path)
    # pred_text = "%s, %.2f" % (results.get("pred_classes"), results.get('scores'))
    print("{}:\n {}".format(test_img_path,results))

    # results = material_pytoch_predict(test_img_path)
    # print("{}:\n {}".format(test_img_path,results))


# if __name__ == "__main__":

    # test()

    # filename = "../demo.jpg"
    # model, categories = get_caffe_model()


    #
    # image = cv2.imread(filename)
    #
    # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    #
    #
    # image = Image.open(filename)
    # img = image.convert('RGB')
    # bgr_img = cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)
    # rgb_img = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2RGB)
    #
    # x1,y1,x2,y2 = 103, 71, 180, 348
    # crop = rgb_img[y1:y2,x1:x2]
    #
    #
    # log = material_caffe_crop_predict(crop)
    # print(log)



