# %%

###### 

from PIL import Image
import cv2, os, datetime, glob,time
import matplotlib

matplotlib.use('TkAgg')
import numpy as np
import tensorflow as tf
from object_detection.utils import label_map_util
from object_detection.utils import visualization_utils as vis_util
from object_detection.utils import ops as utils_ops

from matplotlib import pyplot as plt

start_time = time.time()

def load_image_into_numpy_array(image):
	(im_width, im_height) = image.size
	return np.array(image.getdata()).reshape(
		(im_height, im_width, 3)).astype(np.uint8)


def run_inference_for_single_image(image, graph):
	with graph.as_default():
		with tf.Session() as sess:
			# Get handles to input and output tensors
			ops = tf.get_default_graph().get_operations()
			all_tensor_names = {output.name for op in ops for output in op.outputs}
			tensor_dict = {}
			for key in [
				'num_detections', 'detection_boxes', 'detection_scores',
				'detection_classes', 'detection_masks'
			]:
				tensor_name = key + ':0'
				if tensor_name in all_tensor_names:
					tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(
						tensor_name)
			if 'detection_masks' in tensor_dict:
				# The following processing is only for single image
				detection_boxes = tf.squeeze(tensor_dict['detection_boxes'], [0])
				detection_masks = tf.squeeze(tensor_dict['detection_masks'], [0])
				# Reframe is required to translate mask from box coordinates to image coordinates and fit the image size.
				real_num_detection = tf.cast(tensor_dict['num_detections'][0], tf.int32)
				detection_boxes = tf.slice(detection_boxes, [0, 0], [real_num_detection, -1])
				detection_masks = tf.slice(detection_masks, [0, 0, 0], [real_num_detection, -1, -1])
				detection_masks_reframed = utils_ops.reframe_box_masks_to_image_masks(
					detection_masks, detection_boxes, image.shape[0], image.shape[1])
				detection_masks_reframed = tf.cast(
					tf.greater(detection_masks_reframed, 0.5), tf.uint8)
				# Follow the convention by adding back the batch dimension
				tensor_dict['detection_masks'] = tf.expand_dims(
					detection_masks_reframed, 0)
			image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
			
			# Run inference
			output_dict = sess.run(tensor_dict, feed_dict={image_tensor: np.expand_dims(image, 0)})
			
			# all outputs are float32 numpy arrays, so convert types as appropriate
			output_dict['num_detections'] = int(output_dict['num_detections'][0])
			output_dict['detection_classes'] = output_dict[
				'detection_classes'][0].astype(np.uint8)
			output_dict['detection_boxes'] = output_dict['detection_boxes'][0]
			output_dict['detection_scores'] = output_dict['detection_scores'][0]
			
			if 'detection_masks' in output_dict:
				output_dict['detection_masks'] = output_dict['detection_masks'][0]
	return output_dict


def to_od_model_predict(path_to_labels, test_img_dir, out_root):
	NUM_CLASSES = 2
	IMAGE_SIZE = (12, 8)
	label_map = label_map_util.load_labelmap(path_to_labels)
	categories = label_map_util.convert_label_map_to_categories(label_map,
						max_num_classes=NUM_CLASSES,use_display_name=True)
	category_index = label_map_util.create_category_index(categories)
	
	TestImages = [os.path.join(test_img_dir, file_name) for file_name in os.listdir(test_img_dir)]
	
	for image_path in TestImages:
		img_time_start = time.time()
		image = Image.open(image_path)
		image_np = load_image_into_numpy_array(image)
		# Expand dimensions since the model expects images to have shape: [1, None, None, 3]
		image_np_expanded = np.expand_dims(image_np, axis=0)
		
		# Actual detection.
		output_dict = run_inference_for_single_image(image_np, detection_graph)
		
		# Visualization of the results of a detection.
		vis_util.visualize_boxes_and_labels_on_image_array(
			image_np,
			output_dict['detection_boxes'],
			output_dict['detection_classes'],
			output_dict['detection_scores'],
			category_index,
			instance_masks=output_dict['detection_masks'] if 'detection_masks' in output_dict else None,
			use_normalized_coordinates=True,
			line_thickness=8)
		
		plt.figure(figsize=IMAGE_SIZE)
		plt.imshow(image_np)
		arr = image_path.split('\\')
		arr = arr[-1]
		plt.savefig(out_root + '\\' + arr.split('.')[0] + '_labeled.jpg')
		plt.close()
		img_time_end = time.time()
		
		# time_log = "\t".join([os.path.basename(image_path),str(img_time_end - img_time_start)])
		# print(time_log,file=open(log_file,"a"))
		
		

PATH_TO_MODEL_ROOT = 'E:/HongsenQu/2_project/2_projects/1_od_train/output/'
MODEL_NAME = os.path.join(PATH_TO_MODEL_ROOT,'faster_rcnn_resnet50_coco')
PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')

# MODEL_NAME = r'E:\HongsenQu\2_project\2_projects\1_od_train\mask_rcnn\mask_rcnn_inception_v2_coco'
# PATH_TO_CKPT = os.path.join(MODEL_NAME, 'frozen_inference_graph.pb')

PATH_TO_LABELS = "E:/HongsenQu/2_project/2_projects/images_output/output/new_output/od_data/label.pbtxt"

img_path = "E:/HongsenQu/2_project/2_projects/images_output/output/new_output/od_data/val/JPEGImages"

# out_root = r"E:\HongsenQu\2_project\2_projects\1_od_train\test_result"
out_root = os.path.join(MODEL_NAME,"results")
if not os.path.exists(out_root):os.makedirs(out_root)

if os.path.exists(PATH_TO_CKPT):
	print("exist model !")

detection_graph = tf.Graph()
with detection_graph.as_default():
	od_graph_def = tf.GraphDef()
	with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
		serialized_graph = fid.read()
		od_graph_def.ParseFromString(serialized_graph)
		tf.import_graph_def(od_graph_def, name='')

to_od_model_predict(PATH_TO_LABELS, img_path, out_root)


end_time = time.time()

time_log = "\t".join(['total_time_cost',str(end_time - start_time)])
