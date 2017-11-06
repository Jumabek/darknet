#ifndef REGION_AMBIGUOUS_LAYER_H
#define REGION_AMBIGUOUS_LAYER_H

#include "layer.h"
#include "network.h"

typedef layer region_ambiguous_layer;
region_ambiguous_layer make_region_ambiguous_layer(int batch, int h, int w, int n, int classes, int coords);
void forward_region_ambiguous_layer(const region_ambiguous_layer l, network_state state);
void backward_region_ambiguous_layer(const region_ambiguous_layer l, network_state state);
void get_region_ambiguous_boxes(layer l, int w, int h, float thresh, float **probs, box *boxes, int only_objectness, int *map);
void resize_region_ambiguous_layer(layer *l, int w, int h);

#ifdef GPU
void forward_region_ambiguous_layer_gpu(const region_ambiguous_layer l, network_state state);
void backward_region_ambiguous_layer_gpu(region_ambiguous_layer l, network_state state);
#endif

#endif
