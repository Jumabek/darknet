#include "dts_fire_detector.h"
#include "network.h"
#include "utils.h"
#include "parser.h"
#include "image.h"
#include "opencv2/imgproc/imgproc_c.h"
#include <time.h>
#include <winsock.h>
#include "gettimeofday.h"

extern image ipl_to_image(IplImage* src);
int all_same(int* scores, layer l, int ll_i, int ll_j, int ur_i, int ur_j, int cls_id);
int area(int ll_i, int ll_j, int ur_i, int ur_j);
dts_bbox get_max_bbox(int* scores, layer l, int cls_id);
void print_fire_scores(int* scores, layer l);
void patches_to_rect(int * scores, layer l, dts_fire_detections* detection_results);
void resize_rects(dts_fire_detections* detection_results, int old_w, int old_h, int new_w, int new_h);

static network net;
static image in;
static image in_s;
static image det_s;

static IplImage* input_src;
static float* predictions;
int const FIRE_ID = 0;
int const SMOKE_ID = 1;

double get_wall_time2()
{
	struct timeval time;
	if (gettimeofday(&time, NULL)) {
		return 0;
	}
	return (double)time.tv_sec + (double)time.tv_usec * .000001;
}

//loads the fire_detector once
void load_fire_detector(fire_detector* dtsFireDetector, char* cfgfile, char* weightfile) {
	dtsFireDetector->net = parse_network_cfg(cfgfile);

	if (weightfile) {
		load_weights(&(dtsFireDetector->net), weightfile);
	}
	srand(2222222);
}


//gets called for each frame, returns detection_result
void detect_fire(fire_detector* dtsFireDetector, IplImage *src, dts_fire_detections *detection_result) {

	double before = get_wall_time2();
	double fps;
	input_src = src;

	pthread_t fetch_thread;
	pthread_t detect_thread;
	
	net = dtsFireDetector->net;

	


	dts_bbox* bboxes = detection_result->bboxes;

	in = ipl_to_image(input_src);
	rgbgr_image(in);
	in_s = resize_image(in, net.w, net.h);

	layer l = net.layers[net.n - 1];
	float *X = in_s.data;
	predictions = network_predict(net, X);

	const int background_cls_id = 2;

	int i, j;
	
	int *scores = calloc(l.h*l.w, sizeof(int));

	for (j = 0; j < l.h; j++) {
		for (i = 0; i < l.w; i++) {
			int index = j*l.w*l.classes + i*l.classes;
			int class = max_index(predictions + index, l.classes);
			scores[index / l.classes] = class;		
		}
	}


	patches_to_rect(scores, l,detection_result);
	
	resize_rects(detection_result,net.w,net.h, src->width, src->height);
	double after = get_wall_time2();

	double curr = 1. / (after - before);
	fps = curr;
	printf("\nFPS: %.1f\n", fps);
	free(scores);
	free_image(in);
	free_image(in_s);
}


int get_max_num_fire_det(fire_detector* fireDetector) {
	//calculating how many detections might be returned, 
	//we will allocate memory accordingly from the client side

	network net = fireDetector->net;
	layer l = net.layers[net.n-1];
	return l.w*l.h;
}

void patches_to_rect(int * scores, layer l, dts_fire_detections* detection_results) {
	int count = 0;
	dts_bbox* bboxes = detection_results->bboxes;


	while (1) {
		dts_bbox found_bbox = get_max_bbox(scores, l, FIRE_ID);
		
		if (found_bbox.w <= 0) break;
		bboxes[count++] = found_bbox;
	}

	while (1) {
		dts_bbox found_bbox = get_max_bbox(scores, l, SMOKE_ID);

		if (found_bbox.w <= 0) break;
		bboxes[count++] = found_bbox;
	}

	detection_results->num_detections = count;
	//detection_results->bboxes = bboxes;
}

void resize_rects(dts_fire_detections* detection_results,int old_w, int old_h, int new_w, int new_h) {
	int n = detection_results->num_detections;
	int i;

	dts_bbox* bboxes =  detection_results->bboxes;
	
	for (i = 0; i < n; i++) {
		bboxes[i].xmin = bboxes[i].xmin*new_w / old_w;
		bboxes[i].ymin = bboxes[i].ymin*new_h / old_h;
		bboxes[i].w = bboxes[i].w*new_w / old_w;
		bboxes[i].h = bboxes[i].h*new_h / old_h;
	}
}


int all_same(int* scores, layer l, int ll_i, int ll_j, int ur_i, int ur_j, int cls_id) {
	int i, j;

	for (j = ll_j; j <= ur_j; j++)
		for (i = ll_i; i <= ur_i; i++)
			if (scores[j*l.w + i] != cls_id)
				return 0;
	return 1;
}

int area(int ll_i, int ll_j, int ur_i, int ur_j) {
	return (ur_j - ll_j) * (ur_i - ll_i);
}

dts_bbox get_max_bbox(int* scores, layer l, int cls_id) {
	int best_ll_i = 0, best_ll_j = 0;
	int best_ur_i = -1, best_ur_j = -1;
	int ll_i, ll_j, ur_i, ur_j;

	for (ll_j = 0; ll_j < l.h; ll_j++) {
		for (ll_i = 0; ll_i < l.w; ll_i++) {
			for (ur_j = ll_j; ur_j < l.h; ur_j++) {
				for (ur_i = ll_i; ur_i < l.w; ur_i++) {
					if (area(ll_i, ll_j, ur_i + 1, ur_j + 1) > area(best_ll_i, best_ll_j, best_ur_i + 1, best_ur_j + 1)
						&& all_same(scores, l, ll_i, ll_j, ur_i, ur_j, cls_id)) {
						best_ll_i = ll_i;
						best_ll_j = ll_j;
						best_ur_i = ur_i;
						best_ur_j = ur_j;
					}
				}
			}
		}
	}


	//set class scores inside best rectangle to -1
	int i, j;
	for (j = best_ll_j; j <= best_ur_j; j++) {
		for (i = best_ll_i; i <= best_ur_i; i++) {
			scores[j*l.w + i] = -1;
		}
	}
	dts_bbox max_found_rect = { .cls_id = cls_id,.xmin = best_ll_i * 32,.ymin = best_ll_j * 32,
		.w = (best_ur_i - best_ll_i + 1) * 32,.h = (best_ur_j - best_ll_j + 1) * 32 };
	return max_found_rect;
}

void print_fire_scores(int* scores, layer l) {
	int i, j;
	for (j = 0; j < l.h; j++) {
		for (i = 0; i < l.w; i++) {
			int index = j*l.w + i;
			printf("%d ", scores[index]);
		}
		printf("\n");
	}
	printf("\n");
}
