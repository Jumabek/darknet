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
#define DTS_FIRE_DETECTOR_API __declspec(dllexport)
#else
#define DTS_FIRE_DETECTOR_API __declspec(dllimport)
#endif // DLL_EXPORT


typedef struct fire_detector {
	network net;
} fire_detector;

typedef struct dts_bbox {
	int cls_id;
	float xmin;
	float ymin;
	float w;
	float h;
} dts_bbox;

typedef struct {
	dts_bbox* bboxes;
	int num_detections;
} dts_fire_detections;


DTS_FIRE_DETECTOR_API void load_fire_detector(fire_detector* fireDetector, char* cfg_file, char* weights_file);
DTS_FIRE_DETECTOR_API void detect_fire(fire_detector* fireDetector, IplImage* src, dts_fire_detections* detections);
DTS_FIRE_DETECTOR_API int get_max_num_fire_det(fire_detector* fireDetector);

#ifdef __cplusplus
}
#endif