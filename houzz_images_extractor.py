#coding:utf-8
import os,cv2,glob,h5py
import numpy as np
from numpy import linalg as LA
import matplotlib.pyplot as plt
import torchvision.models as models
from torch.autograd import Variable
import torchvision.transforms as transforms
import time


model_vgg19 = models.vgg19(pretrained=True)

def feature_extractor(modelName, file_path):
    img_size = 224
    img_to_tensor = transforms.ToTensor()

    # if modelName=="resnet18":
    #     model = models.resnet18(pretrained=True)

    # elif modelName=="resnet50":
    #     model = models.resnet50(pretrained=True)

    # elif modelName=="resnet101":
    #     model = models.resnet101(pretrained=True)

    # elif modelName == "resnet152":
    #     model = models.resnet152(pretrained=True)

    # elif modelName == "vgg11":
    #     model = models.vgg11(pretrained=True)

    # elif modelName == "vgg16":
    #     model = models.vgg16(pretrained=True)

    # elif modelName == "vgg19":
    #     model = models.vgg19(pretrained=True)

    # elif modelName == "densenet121":
    #     model = models.densenet121(pretrained=True)

    # elif modelName == "densenet161":
    #     model = models.densenet161(pretrained=True)

    # elif modelName == "inception_v3":
    #     model = models.inception_v3(pretrained=True)

    # else:
    #     print("model to use:{}".format("resnet18","resnet50","resnet101","resnet152","vgg11", "vgg16", "vgg19","densenet121", "densenet161","inception_v3"))

    model = model_vgg19
    for param in model.parameters():
        param.requires_grad = False
    model.eval()

    # model = model.cuda()

    src = cv2.imread(file_path)
    name = os.path.basename(file_path)
    img_resized = cv2.resize(src, (img_size, img_size))
    img = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB)
    img_tensor = img_to_tensor(img).unsqueeze(0)
    output = Variable(img_tensor.float(), requires_grad=False)

    feature = model(output).cpu()
    feature = feature.data.numpy()

    feat = feature[0]
    norm_feat = feat / LA.norm(feat)

    return name, norm_feat


def load_houzz_images():
    img_type = [".jpg",".png",".jpeg",'.bmp','.tif']
    houzz_root = "/data/dadi/houzz_5space"

    houzz_imgs = []
    image_names = []

    for category in os.listdir(houzz_root):
        img_dir = os.path.join(houzz_root,category)
        if os.path.isdir(img_dir):

            for file_name in os.listdir(img_dir):
                file_type = os.path.splitext(file_name)[-1]
                img_path = os.path.join(img_dir,file_name)

                if file_type in img_type:
                    if file_name not in image_names:
                        image_names.append(file_name)
                        houzz_imgs.append(img_path)

                    # else:
                    #     print(img_path)

    return houzz_imgs


def get_each_category_houzz_images():
    houzz_root = "/data/dadi/houzz_5space"
    img_type = [".jpg",".png",".jpeg",'.bmp','.tif']


    image_dic = {}

    for category in os.listdir(houzz_root):
        img_dir = os.path.join(houzz_root,category)
        category_imgs = []
        if os.path.isdir(img_dir):
            for file_name in os.listdir(img_dir):
                file_type = os.path.splitext(file_name)[-1]
                img_path = os.path.join(img_dir,file_name)
                if file_type in img_type:
                    category_imgs.append(img_path)

        image_dic[category] = category_imgs

    return image_dic



def creat_images_feat_db(imageList,modelName):

    save_dir = "/home/user/tmp/pycharm_project_310/1_detectron2/ImageDetectionAPI/image_similarity_recognition/houzz_5space/src"
    saveFeatsFile = os.path.join(save_dir,'houzz_5space_{}_feat.pt'.format(modelName))

    if not os.path.exists(saveFeatsFile):
        extract_start = time.time()

        print("***********  {} start image-feature extraction ...".format(modelName))
        names,feats = [],[]
        n = 0
        for img_path in imageList:
            if os.path.exists(img_path):
                n+=1
                print("modelName: {}, image_feature_extract: {} , {}/{}".format(modelName, img_path, n , len(imageList)))
                name, feat = feature_extractor(modelName, img_path)
                names.append(name)
                feats.append(feat)

        feats = np.array(feats)

        print("modelName: writing ....")
        h5f = h5py.File(saveFeatsFile, 'w')
        h5f.create_dataset('dataset_1', data=feats)
        h5f.create_dataset('dataset_2', data=np.string_(names))
        h5f.close()

        extract_end = time.time()

        extract_cost = extract_end - extract_start

        if os.path.exists(saveFeatsFile):
            print("modelName:{} features saved:\n{}".format(modelName, saveFeatsFile))

        print("################  {}  image-feature extraction takes {}".format(modelName,extract_cost))


    else:
        print('feature_db exits: \n {}'.format(saveFeatsFile))


def creat_images_feat_db_v1(imageList,prefix,modelName):

    save_dir = "/home/user/tmp/pycharm_project_310/1_detectron2/ImageDetectionAPI/image_similarity_recognition/houzz_5space/src"
    saveFeatsFile = os.path.join(save_dir,'houzz_5space_{}_{}_feat.pt'.format(prefix,modelName))

    if not os.path.exists(saveFeatsFile):
        extract_start = time.time()

        print("***********  {} start image-feature extraction ...".format(modelName))
        names,feats = [],[]
        n = 0
        for img_path in imageList:
            if os.path.exists(img_path):
                n+=1
                print("modelName: {}, image_feature_extract: {} , {}/{}".format(modelName, img_path, n , len(imageList)))
                name, feat = feature_extractor(modelName, img_path)
                names.append(name)
                feats.append(feat)

        feats = np.array(feats)

        print("modelName: writing ....")
        h5f = h5py.File(saveFeatsFile, 'w')
        h5f.create_dataset('dataset_1', data=feats)
        h5f.create_dataset('dataset_2', data=np.string_(names))
        h5f.close()

        extract_end = time.time()

        extract_cost = extract_end - extract_start

        if os.path.exists(saveFeatsFile):
            print("modelName:{} features saved:\n{}".format(modelName, saveFeatsFile))

        print("################  {}  image-feature extraction takes {}".format(modelName,extract_cost))


    else:
        print('feature_db exits: \n {}'.format(saveFeatsFile))



def houzz_feat_db():

    model_list = ["resnet18","resnet50","resnet101","resnet152","vgg11", "vgg16", "vgg19","densenet121", "densenet161","inception_v3"]

    imageList = load_houzz_images()
    for modelName in model_list:
        creat_images_feat_db(imageList,modelName)


def houzz_feat_db_v1():

    model_list = ["resnet50","vgg11", "vgg16", "vgg19"]
    image_dic = get_each_category_houzz_images()
    for category in image_dic:
        category_images = image_dic.get(category)
        print(category_images)
        prefix = category
        for modelName in model_list:
            print("each_category_extrac: {}, {} ...".format(category, modelName))
            creat_images_feat_db_v1(category_images,prefix, modelName)




def extract_feat_test():
    test_path = "/data/dadi/houzz_5space/bath/1f41790808934aa0_9-3281.jpg"
    modelName = 'vgg16'
    modelName = 'resnet152'
    name, feat = feature_extractor(modelName,test_path)

    print("image: {},modelName: {},  feat.shape:{}".format(name, modelName,feat.shape))





if __name__ == "__main__":
    extract_feat_test()
    # houzz_feat_db()

    # image_dic = get_each_category_houzz_images()
    # print(image_dic)
    #
    # for cat in image_dic:
    #     print(cat, len(image_dic.get(cat)))

    # houzz_feat_db_v1()

