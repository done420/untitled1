####  label me output jsons(to coco jsons) check

import os,cv2,json,sys

# json_file = "E:\\HongsenQu\\2_project\\0_test\\1_detectron2_test\\furniture-detector\\val\\my_test_data\\my_test.json"
# images_dir = "E:\\HongsenQu\\2_project\\0_test\\1_detectron2_test\\furniture-detector\\val\\my_test_data\\images"

def load_coco_json(json_file,images_dir):
	with open(json_file) as f:
		print('loading json file:')
		data = json.load(f)
		print('loading done.')
		images = data['images']
		categories = data['categories']
		annotations = data['annotations']
		imgages_name = images[0]['file_name']
		
		image_dict = dict([int(images[i]['id']), images[i]['file_name']] for i in range(len(images)))
		categorie_dict = dict([int(categories[i]['id']), categories[i]['name']] for i in range(len(categories)))
		annotation_seg_dict = dict([int(annotations[i]['image_id']), annotations[i]['segmentation']] for i in range(len(annotations)))
		annotation_bbox_dict = dict([int(annotations[i]['image_id']), annotations[i]['bbox']] for i in range(len(annotations)))
		category_id_dict = dict([int(annotations[i]['image_id']), annotations[i]['category_id']] for i in range(len(annotations)))
		
		print("num annotations: %d" % len(annotations))
		#     print(file_list)
		
		for image_id in image_dict:
			file_name = image_dict.get(image_id)
			coco_type = "COCO_val2014_"
			image_path = os.path.join(images_dir, str(image_dict.get(image_id)))
			
			if os.path.exists(image_path):
				print(image_path)
				category_id = category_id_dict.get(image_id)
				category_name = categorie_dict.get(category_id)
				
				bbox = annotation_bbox_dict.get(image_id)  ## x,y,h,w
				# print(bbox)
				
				rois = annotation_seg_dict.get(image_id)[0] # [[x1,y1,x2,y2, ....]] to [x1,y1,x2,y2, ....]
				x1, y1, w, h = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
				x2, y2 = x1 + w, y1 + h
				
				img = cv2.imread(image_path,-1)
				cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)  ##
				
				for index in range(0, len(rois),2):
					x,y = (int(rois[index]), int(rois[index + 1]))
					cv2.circle(img,(x,y),2, (0,255,0), 5)
					
				wind_name = "%s : %s"%(category_name,file_name)
				cv2.namedWindow(wind_name,0)
				cv2.imshow(wind_name, img)
				cv2.waitKey(0)
			else:
				print("No file: %s" % image_path)
				# break


coco_json_file  = sys.argv[1]
coco_images_dir = sys.argv[2]

load_coco_json(coco_json_file,coco_images_dir)



