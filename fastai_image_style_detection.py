####coding: utf-8

import fastai,cv2,os,sys
from fastai import *
from fastai.vision import *
import torch


def fastai_detection(test_img_in):

    defaults.device = torch.device('cpu')

    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    # print("father_path:",father_path)

    model_root = os.path.join(father_path, "./models/")

    if os.path.exists(os.path.join(model_root,"export.pkl")) and os.access(model_root, os.R_OK):
        print("model load ....")
    else:
        print("NotFound:{}".format(os.path.join(model_root,"export.pkl")))


    learn = load_learner(path = model_root, file="export.pkl")
    learn = to_cpu(learn)

    classes = learn.data.classes

    img = open_image(test_img_in)


    cat,pred_idx,probs = learn.predict(img)
    probs = probs.numpy().tolist()

    pred_idx = pred_idx.numpy().tolist()
    pred_score = probs[pred_idx]
    pred_class = classes[pred_idx]


    result = {
        "score": pred_score,### 预测分值，0.99
        "pred_class": pred_class,### 预测物体编码，dizhonghai
            }

    # print(result)
    return result



def load_style():
    defaults.device = torch.device('cpu')

    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")
    # print("father_path:",father_path)

    model_root = os.path.join(father_path, "./models/")

    if os.path.exists(os.path.join(model_root,"export.pkl")) and os.access(model_root, os.R_OK):
        print("model load ....")
    else:
        print("NotFound:{}".format(os.path.join(model_root,"export.pkl")))

    learn = load_learner(path = model_root, file="export.pkl")
    learn = to_cpu(learn)
    classes = learn.data.classes
    return learn, classes


LEARN, LEARN_CLASSES = load_style()


def style_predict(test_img_in):

    img = open_image(test_img_in)

    cat,pred_idx,probs = LEARN.predict(img)
    probs = probs.numpy().tolist()

    pred_idx = pred_idx.numpy().tolist()
    pred_score = probs[pred_idx]
    pred_class = LEARN_CLASSES[pred_idx]


    result = {
        "score": pred_score,### 预测分值，0.99
        "pred_class": pred_class,### 预测物体编码，dizhonghai
    }

    return result



def test():

    test_img_path = "../demo.jpg"
    # result = fastai_detection(test_img_path)
    result = style_predict(test_img_path)
    print(result)

# if __name__ == "__main__":
#     test()