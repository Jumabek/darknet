#ifndef PATH_REGION_LAYER_H
#define PATH_REGION_LAYER_H

#include "layer.h"
#include "network.h"

typedef layer patch_region_layer;

patch_region_layer make_patch_region_layer(int batch, int h, int w, int classes);
void forward_patch_region_layer(const patch_region_layer l, network_state state);
void backward_patch_region_layer(const patch_region_layer l, network_state state);
void get_patch_region_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map);
void resize_patch_region_layer(layer *l, int w, int h);

#ifdef GPU
void forward_patch_region_layer_gpu(const patch_region_layer l, network_state state);
void backward_patch_region_layer_gpu(patch_region_layer l, network_state state);
#endif

#endif
