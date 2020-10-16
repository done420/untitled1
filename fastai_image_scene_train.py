##coding:utf-8
import os,sys
from fastai import *
from fastai.vision import *
from fastai.vision.learner import cnn_learner
from fastai.vision import ImageDataBunch, imagenet_stats, get_transforms
from fastai.train import ClassificationInterpretation


def train(imgs_root,model_dir):

    my_tfms=get_transforms()
    print(f"Transforms on Train set: {my_tfms[0]}")
    print(f"Transforms on Validation set: {my_tfms[1]}")

    np.random.seed(42)
    ### '/home/user/tmp/pycharm_project_310/1_detectron2/Furniture-Style-Classifier-master/Data'

    # imgs_root = '/home/user/tmp/pycharm_project_310/1_detectron2/Furniture-Style-Classifier-master/MyData'

    data = ImageDataBunch.from_folder(path= Path(imgs_root),
                                      train=".",
                                      valid_pct=0.2,
                                      ds_tfms = my_tfms,
                                      size=224,
                                      num_workers=4,
                                      bs = 64).normalize(imagenet_stats)

    print(f"BatchSize: {data.batch_size}")
    print(f"Train Dataset size: {len(data.train_ds)}")
    print(f"Validataion Dataset size: {len(data.valid_ds)}")
    print(f"Classes: {data.classes}")
    print(f"Number of Classes : {data.c}")

    num_epochs = 5
    lr = 4.37E-03
    learn = cnn_learner(data,models.resnet34, metrics=error_rate, pretrained=True,true_wd = False, train_bn = False)
    learn.fit(epochs = num_epochs,lr=lr)

    #### 模型评估
    report = learn.interpret()
    matrix = report.confusion_matrix().tolist()

    print("confusion_matrix:\n{}".format(matrix))

    learn.model = learn.model.cpu() ### 转化为cpu模型

    # model_dir = os.path.join(os.getcwd(),"./models")
    # model_dir = '/home/user/tmp/pycharm_project_310/1_detectron2/ImageDetectionAPI/image_style_classifier/models/'

    weight_path = os.path.join(model_dir,'resnet34_scene_detection')
    inference_path = os.path.join(model_dir,'export.pkl')

    learn.save(weight_path)
    learn.export(file = Path(inference_path))

    if os.path.exists(inference_path):
        print("model save to :{}".format(model_dir))


if __name__ == "__main__":

    imgs_root = sys.argv[1] ## 图像路径父目录（绝对路径）
    model_dir = sys.argv[2] ## 模型保存路径（绝对路径）

    train(imgs_root,model_dir)
