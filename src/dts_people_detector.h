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
#define 	DTS_PEOPLE_DETECTOR_API __declspec(dllexport)
#else
#define DTS_PEOPLE_DETECTOR_API __declspec(dllimport)
#endif // DTS_PEOPLE_DETECTOR_EXPORTS

typedef struct people_detector{
	network net;
	float thresh;
} people_detector;

typedef struct dts_rect {
	float xmin;
	float ymin;
	float w;
	float h;
} dts_rect;


typedef struct dts_people_detections {
	dts_rect* rects;
	float* probabilities;
	int num_of_detections;
} dts_people_detections;


DTS_PEOPLE_DETECTOR_API void load_detector(people_detector* peopleDetector, char* cfg_file, char* weights_file, float thresh);
DTS_PEOPLE_DETECTOR_API void detect_people(people_detector* peopleDetector, IplImage* src, dts_people_detections* detections);
DTS_PEOPLE_DETECTOR_API int get_max_num_det(people_detector* peopleDetector);

#ifdef __cplusplus
}
#endif