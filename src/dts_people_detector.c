#include "dts_people_detector.h"
#include "network.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "opencv2/imgproc/imgproc_c.h"

extern image ipl_to_image(IplImage* src);

//loads the peole_detector once
void load_detector(people_detector* dtsPeopleDetector, char* cfgfile, char* weightfile, float thresh ) {
	dtsPeopleDetector->net = parse_network_cfg(cfgfile);
	dtsPeopleDetector->thresh = thresh;

	if (weightfile) {
		load_weights(&(dtsPeopleDetector->net), weightfile);
	}
	srand(2222222);
}


//gets called for each frame, returns detection_result
void detect_people(people_detector* dtsPeopleDetector, IplImage *src, dts_people_detections *detection_result) {

	double before = get_wall_time();
	double fps;

	network net = dtsPeopleDetector->net;
	float thresh = dtsPeopleDetector->thresh;

	int j;
	float nms = .4;


	image im = ipl_to_image(src);
	rgbgr_image(im);
	image sized = resize_image(im, net.w, net.h);

	layer l = net.layers[net.n - 1];


	box *boxes = (box *)calloc(l.w*l.h*l.n, sizeof(box));
	float **probs = (float **)calloc(l.w*l.h*l.n, sizeof(float *));
	for (j = 0; j < l.w*l.h*l.n; ++j) probs[j] = (float*)calloc(l.classes, sizeof(float *));

	float *X = sized.data;

	network_predict(net, X);


	get_region_boxes(l, im.w, im.h, thresh, probs, boxes, 0, 0);
	if (nms) do_nms_sort(boxes, probs, l.w*l.h*l.n, l.classes, nms);



	float* probabilities = detection_result->probabilities;
	dts_rect* rects = detection_result->rects;

	int person_ind_in_voc = 14;
	int detection_count = 0;
	int i;
	if (l.classes == 20) {
		for (i = 0; i < l.w*l.h*l.n; i++) {
			float prob = probs[i][person_ind_in_voc];
			if (prob > thresh) {

				//since for people detection we have only one class (which is person), we are extracting probabilities only for person
				probabilities[detection_count] = prob;

				float xmin = boxes[i].x - boxes[i].w / 2.;
				float ymin = boxes[i].y - boxes[i].h / 2.;
				float width = boxes[i].w;
				float height = boxes[i].h;

				if (xmin < 0) xmin = 0;
				if (ymin < 0) ymin = 0;

				rects[detection_count].xmin = xmin;
				rects[detection_count].ymin = ymin;
				rects[detection_count].w = width;
				rects[detection_count].h = height;
				detection_count++;
			}
		}
	}
	else if(l.classes==1) {
		for (i = 0; i < l.w*l.h*l.n; i++) {
			float prob = probs[i][0];
			if (prob > thresh) {

				//since for people detection we have only one class (which is person), we are extracting probabilities only for person
				probabilities[detection_count] = prob;

				float xmin = boxes[i].x - boxes[i].w / 2.;
				float ymin = boxes[i].y - boxes[i].h / 2.;
				float width = boxes[i].w;
				float height = boxes[i].h;

				if (xmin < 0) xmin = 0;
				if (ymin < 0) ymin = 0;

				rects[detection_count].xmin = xmin;
				rects[detection_count].ymin = ymin;
				rects[detection_count].w = width;
				rects[detection_count].h = height;
				detection_count++;
			}
		}
	}
	else {
		printf("For people detection,\n Number of classes should be either 20 or 1\n");
		return;
	}

	

	detection_result->num_of_detections = detection_count;
	double after = get_wall_time();

	float curr = 1. / (after - before);
	fps = curr;
	before = after;
	printf("FPS = %f\t curr = %f\n", fps, curr);

	free_image(im);
	free_image(sized);
	free(boxes);
	free_ptrs((void **)probs, l.w*l.h*l.n);
}


int get_max_num_det(people_detector* peopleDetector) {
	//calculating how many detections might be returned, 
	//we will allocate memory accordingly from the client side

	network net = peopleDetector->net;
	layer l = net.layers[net.n-1];
	return l.w*l.h*l.n;
}