#!/usr/bin/env python3

import numpy
import glob,cv2,os.path
import caffe
from PIL import Image


caffe.set_mode_gpu()


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


MODEL, CATEGORIES = get_caffe_model()

def material_caffe_crop_predict(rgb_crop):


    im = rgb_crop/255.0
    im = im.astype(numpy.float32)

    output_prob = MODEL.predict([im*255.0])[0]
    # print(output_prob)
    label_index = output_prob.argmax()
    label_name = CATEGORIES[label_index]
    pred_score = output_prob[label_index]

    results = {
        "scores": pred_score,### 预测分值列表，[0.2, ...]
        "pred_classes": label_name,### 预测物体编码列表，['kitchen', ...]
    }


    return results





def test():
    # test_img_path = "../demo.jpg"
    test_img_path = "/home/user/tmp/pycharm_project_310/1_detectron2/ImageDetectionAPI/demo.jpg"

    results = material_caffe_predict(test_img_path)
    # pred_text = "%s, %.2f" % (results.get("pred_classes"), results.get('scores'))
    print("aaaaaaaaaaaaa :{}:\n {}".format(test_img_path,results))

    # results = material_pytoch_predict(test_img_path)
    # print("{}:\n {}".format(test_img_path,results))


# if __name__ == "__main__":
#
#     # test()
#
#     filename = "../demo.jpg"
#     model, categories = get_caffe_model()
#
#     a = [[133.9888458251953, 64.25130462646484, 188.90753173828125, 318.81671142578125], [87.95594024658203, 134.41082763671875, 155.930419921875, 346.4548645019531], [89.28762817382812, 165.3110809326172, 129.28958129882812, 348.48760986328125], [103.83251953125, 71.37813568115234, 180.18321228027344, 348.2173767089844]]
#     #
#     # image = cv2.imread(filename)
#     #
#     # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
#     #
#     #
#     image = Image.open(filename)
#     img = image.convert('RGB')
#     bgr_img = cv2.cvtColor(numpy.asarray(image),cv2.COLOR_RGB2BGR)
#     rgb_img = cv2.cvtColor(bgr_img,cv2.COLOR_BGR2RGB)
#
#     # x1,y1,x2,y2 = 103, 71, 180, 348
#     # crop = rgb_img[y1:y2,x1:x2]
#     # log = material_caffe_crop_predict(crop)
#     # print(log)
#
#     for i in a:
#         x1,y1,x2,y2 = i
#
#         x1 = int(x1)
#         x2 = int(x2)
#         y1 = int(y1)
#         y2 = int(y2)
#         crop = rgb_img[y1:y2,x1:x2]
#         log = material_caffe_crop_predict(crop)
#         print(x1,y1,x2,y2, ": ---->  ", log)




