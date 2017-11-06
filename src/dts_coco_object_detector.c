#include "dts_coco_object_detector.h"
#include "network.h"
#include "utils.h"
#include "parser.h"
#include "image.h"
#include "opencv2/imgproc/imgproc_c.h"
#include <time.h>
#include <winsock.h>
#include "gettimeofday.h"

extern image ipl_to_image(IplImage* src);

static network net;
static image in;
static image in_s;
static image det_s;

static IplImage* input_src;
static float* predictions;

//loads the fire_detector once
void load_coco_object_detector(dts_coco_object_detector* cocoObjectDetector, char* cfgfile, char* weightfile, float thresh) {
	cocoObjectDetector->net = parse_network_cfg(cfgfile);
	cocoObjectDetector->thresh = thresh;
	if (weightfile) {
		load_weights(&(cocoObjectDetector->net), weightfile);
	}
	srand(2222222);
}


//gets called for each frame, returns detection_result
void detect_coco_object(dts_coco_object_detector* cocoObjectDetector, IplImage *src, dts_coco_object_detections *detection_result, int cls_mode) {

	double before = get_wall_time2();
	double fps;

	network net = cocoObjectDetector->net;
	float thresh = cocoObjectDetector->thresh;

	int j;
	float nms = .4;


	image im = ipl_to_image(src);
	rgbgr_image(im);
	image sized = resize_image(im, net.w, net.h);

	layer l = net.layers[net.n - 1];
	l.cls_mode = cls_mode; //this line tells region_layer.get_region_boxes() which classes to detect

	box *boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
	float **probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
	for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)calloc(l.classes, sizeof(float *));

	float *X = sized.data;

	network_predict(net, X);

	get_region_boxes(l, im.w, im.h, thresh, probs, boxes, 0, 0);
	if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);

	dts_bbox* dtsBBoxes = detection_result->bboxes;

	int detection_count = 0;
	int i;
	for (i = 0; i < l.w*l.h*l.n; i++) {		
		int class = max_index(probs[i], l.classes);
		float prob = probs[i][class];
		if (prob > thresh) {

			float xmin = boxes[i].x - boxes[i].w / 2.;
			float ymin = boxes[i].y - boxes[i].h / 2.;
			float width = boxes[i].w;
			float height = boxes[i].h;

			if (xmin < 0) xmin = 0;
			if (ymin < 0) ymin = 0;

			dtsBBoxes[detection_count].probability = prob;
			dtsBBoxes[detection_count].xmin = xmin;
			dtsBBoxes[detection_count].ymin = ymin;
			dtsBBoxes[detection_count].w = width;
			dtsBBoxes[detection_count].h = height;
			dtsBBoxes[detection_count].cls_id = class;
			detection_count++;
		}
	}



	detection_result->num_detections = detection_count;
	double after = get_wall_time2();

	float curr = 1. / (after - before);
	fps = curr;
	before = after;
	printf("FPS = %f\t curr = %f\n", fps, curr);

	free_image(im);
	free_image(sized);
	free(boxes);
	free_ptrs((void **)probs, l.w*l.h*l.n);
}


int get_max_num_coco_object_det(dts_coco_object_detector* cocoObjectDetector) {
	//calculating how many detections might be returned, 
	//we will allocate memory accordingly from the client side

	network net = cocoObjectDetector->net;
	layer l = net.layers[net.n-1];
	return l.w*l.h*l.n;
}
