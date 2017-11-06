#ifndef DEMO
#define DEMO

#include "image.h"
void demo(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, char **names, int classes, int frame_skip, char *prefix);

void demo_patch_classifier(char *cfgfile, char *weightfile, float thresh, int cam_index, const char *filename, int classes, int frame_skip, char *prefix);
#endif
