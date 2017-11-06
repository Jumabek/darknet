#pragma once
#ifdef __cplusplus
extern "C" {
#endif // __cplusplus


#include "network.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "image.h"
#include "utils.h"

#include "opencv2/imgproc/imgproc_c.h"

#ifdef DLL_EXPORT
#define DTS_COCO_OBJECT_DETECTOR_API __declspec(dllexport)
#else
#define DTS_COCO_OBJECT_DETECTOR_API __declspec(dllimport)
#endif // DLL_EXPORT


typedef struct dts_coco_object_detector {
	network net;
	float thresh;
} dts_coco_object_detector;

typedef struct dts_bbox {
	int cls_id;
	float probability;
	float xmin;
	float ymin;
	float w;
	float h;
} dts_bbox;

typedef struct {
	dts_bbox* bboxes;
	int num_detections;
} dts_coco_object_detections;


DTS_COCO_OBJECT_DETECTOR_API void load_coco_object_detector(dts_coco_object_detector* cocoObjectDetector, char* cfg_file, char* weights_file, float thresh);
DTS_COCO_OBJECT_DETECTOR_API void detect_coco_object(dts_coco_object_detector* cocoObjectDetector, IplImage* src, dts_coco_object_detections* detections, int cls_mode);
DTS_COCO_OBJECT_DETECTOR_API int get_max_num_coco_object_det(dts_coco_object_detector* cocoObjectDetector);

#ifdef __cplusplus
}
#endif