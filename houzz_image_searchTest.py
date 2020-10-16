#coding:utf-8
import argparse
import os,cv2,glob,h5py,time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from houzz_images_extractor import feature_extractor
from annoy import AnnoyIndex

db_path = '/home/user/tmp/pycharm_project_310/1_detectron2/ImageDetectionAPI/image_similarity_recognition/houzz_5space/db/houzz_5space_vgg19_feat.pt'


def load_db(db_path):

    try:
        h5f = h5py.File(db_path,'r')
        feats = h5f['dataset_1'][:]
        imgNames = h5f['dataset_2'][:]
        h5f.close()

        new_feats = [[i.tolist()] for i in feats]  ### 1150*1000 to 1150*1
        new_name = [[str(i.tolist(), 'utf-8')] for i in imgNames]

        ### array to Dataframe
        df_feats = pd.DataFrame(new_feats)
        df_name = pd.DataFrame(new_name)
        df_feats = df_feats.rename(columns={0: "img_vector"})  ### 修改列名称
        df_name = df_name.rename(columns={0: "img_name"})
        df = pd.concat([df_name, df_feats], join='inner', axis=1)  ### 合并为一个

        return df
    except:
        print("image database loading error !")
        return None


import time
start_time = time.time()
DATA = load_db(db_path)
index_file = '/home/user/tmp/pycharm_project_310/1_detectron2/ImageDetectionAPI/image_similarity_recognition/space_index.ann'
INDEXER = AnnoyIndex(1000, metric='euclidean')
INDEXER.load(index_file)

def imgage_search(img_path,modelName='vgg19',prefix=None):
    start_time = time.time()
    print('提取图片相似度 -> 开始')
    _, vector = feature_extractor(modelName, img_path)
    print('提取图片相似度 -> 完成', time.time() - start_time)

    start_time = time.time()
    ids = INDEXER.get_nns_by_vector(vector, 10)
    similar_images_df = DATA.iloc[ids]
    similar_image = [str(image_name) for image_name in similar_images_df['img_name']]
    return similar_image


def imgage_index(img_path,modelName,prefix=None):
    current_path = os.path.abspath(__file__)
    father_path = os.path.abspath(os.path.dirname(current_path) + os.path.sep + ".")

    if prefix:
        db_path = os.path.join(father_path,'./db/houzz_5space_{}_{}_feat.pt'.format(prefix,modelName))
    else:
        db_path = os.path.join(father_path,'./db/houzz_5space_{}_feat.pt'.format(modelName))

    if os.path.exists(db_path):

        df = load_db(db_path)

        query_raw_name, query_raw_feat = feature_extractor(modelName, img_path)

        query_feat_annoy = [query_raw_feat.tolist()]
        query_new_name = str(query_raw_name) + "_query"
        df_query_name = pd.DataFrame({"img_name": [query_new_name]})
        df_query_feat = pd.DataFrame({"img_vector": query_feat_annoy})
        df_query = pd.concat([df_query_name, df_query_feat], join='inner', axis=1)

        df_merge = pd.concat([df, df_query], ignore_index=True)
        # print(df_merge)

        qury_index = df_merge[(df_merge.img_name == query_new_name)].index.tolist()[0]


        #### 构建annoy
        f = len(df_merge['img_vector'][0])
        t = AnnoyIndex(f, metric='euclidean')

        if prefix:
            ntree = 100
        else:
            ntree = 5 ## bath  bedroom  dining  kitchen  living

        num_similar_images = 10
        for i, vector in enumerate(df_merge['img_vector']):
            t.add_item(i, vector)
        _ = t.build(ntree)

        t.save('/home/user/tmp/pycharm_project_310/1_detectron2/ImageDetectionAPI/image_similarity_recognition/space_index.ann')

        def get_similar_images_annoy(img_index):
            base_img_id, base_vector, base_label = df_merge.iloc[img_index, [0, 1, 0]]
            similar_img_ids = t.get_nns_by_item(img_index, num_similar_images)
            return base_img_id, base_label, df_merge.iloc[similar_img_ids]

        test_img_index = qury_index
        image_name, base_label, similar_images_df = get_similar_images_annoy(test_img_index)

        #### 相似图像
        similar_image = [str(image_name) for image_name in similar_images_df['img_name'] if str(image_name) != str(query_new_name)]
        # print(similar_image)

        return similar_image



def do_predict():

    parser = argparse.ArgumentParser(description='Similar Images Search.')
    parser.add_argument('-image_input',metavar='image_path',  help = "path to the image")
    parser.add_argument('-model_name',metavar='model',type= str, default='vgg19')
    parser.add_argument('-prefix',metavar='prefix', type= str, default='None',choices=['None','bath','bedroom', 'dining', 'kitchen', 'living'],help="select a category, use None as  default.")

    args = parser.parse_args()
    # ret = vars(args)
    # print(type(ret))
    # print(ret)

    result = imgage_search(args.image_input,args.model_name)
    # print(result)

    return result

def test():
    modelName = 'vgg19'
    img_path = "/data/dadi/houzz_5space/kitchen/a7e12aa207c46a3e_9-0484.jpg"
    prefix = 'kitchen'

    result = imgage_search(img_path,modelName,prefix)



if __name__=="__main__":

    # result = test()
    # result = do_predict()
    # print(result)

    img_path = '/data/dadi/houzz_5space/living/ffe1e7c6086be36c_9-5086.jpg'
    modelName = 'vgg19'
    prefix = ''
    print('1')
    result = imgage_search(img_path,modelName,prefix)
    print(result)
    print('2')


