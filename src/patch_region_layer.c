#include "patch_region_layer.h"
#include "activations.h"
#include "blas.h"
#include "box.h"
#include "cuda.h"
#include "utils.h"
#include <stdio.h>
#include <assert.h>
#include <string.h>
#include <stdlib.h>

#define DOABS 1

patch_region_layer make_patch_region_layer(int batch, int w, int h, int classes)
{
	patch_region_layer l = { 0 };
	l.type = PATCH_REGION;
	l.batch = batch;
	l.h = h;
	l.w = w;
	l.classes = classes;
	l.cost = calloc(1, sizeof(float));
	l.outputs = h*w*(classes);
	l.inputs = l.outputs;
	l.delta = calloc(batch*l.outputs, sizeof(float));
	l.output = calloc(batch*l.outputs, sizeof(float));
	int i;
	

	l.forward = forward_patch_region_layer;
	l.backward = backward_patch_region_layer;
#ifdef GPU
	l.forward_gpu = forward_patch_region_layer_gpu;
	l.backward_gpu = backward_patch_region_layer_gpu;
	l.output_gpu = cuda_make_array(l.output, batch*l.outputs);
	l.delta_gpu = cuda_make_array(l.delta, batch*l.outputs);
#endif

	fprintf(stderr, "detection\n");
	srand(0);

	return l;
}

void patch_resize_region_layer(layer *l, int w, int h)
{
	l->w = w;
	l->h = h;

	l->outputs = h*w*l->classes;
	l->inputs = l->outputs;

	l->output = realloc(l->output, l->batch*l->outputs * sizeof(float));
	l->delta = realloc(l->delta, l->batch*l->outputs * sizeof(float));

#ifdef GPU
	cuda_free(l->delta_gpu);
	cuda_free(l->output_gpu);

	l->delta_gpu = cuda_make_array(l->delta, l->batch*l->outputs);
	l->output_gpu = cuda_make_array(l->output, l->batch*l->outputs);
#endif
}




void forward_patch_region_layer(const patch_region_layer l, network_state state)
{
	if (!state.truth) return;
	int i, j, b;

	//printf("state.input[0] = %d", state.input[0]);
	//printf("l.output[0] = %f\n", l.output[0]);
	//printf("l.batch = %d\n", l.batch);
	//printf("state.input[0] = %f\n", state.input[0]);
	memcpy(l.output, state.input, l.outputs*l.batch * sizeof(float));
	#ifndef GPU
	flatten(l.output, l.w*l.h, size, l.batch, 1);
	#endif
	for (b = 0; b < l.batch; ++b) {
		for (i = 0; i < l.h*l.w; ++i) {
			int index = l.classes*i + b*l.outputs;
			softmax(l.output + index, l.classes, 1, l.output + index);
		}
	}
	
	if (!state.train) return;
	memset(l.delta, 0, l.outputs*l.batch * sizeof(float));
	*(l.cost) = 0;

	


	l2_cpu_batch(l.batch,l.outputs, l.output, state.truth, l.delta, l.output);

	l.cost[0] = sum_array_batch(l.output, l.inputs, l.batch);
}

void backward_patch_region_layer(const patch_region_layer l, network_state state)
{
	axpy_cpu(l.batch*l.inputs, 1, l.delta, 1, state.delta, 1);
}



#ifdef GPU

void forward_patch_region_layer_gpu(const patch_region_layer l, network_state state)
{

	flatten_ongpu(state.input, l.h*l.w, l.classes, l.batch, 1, l.output_gpu);

	float *in_cpu = calloc(l.batch*l.inputs, sizeof(float));
	float *truth_cpu = 0;
	if (state.truth) {
		int num_truth = l.batch*l.outputs;
		truth_cpu = calloc(num_truth, sizeof(float));
		cuda_pull_array(state.truth, truth_cpu, num_truth);
	}
	cuda_pull_array(l.output_gpu, in_cpu, l.batch*l.inputs);
	network_state cpu_state = state;
	cpu_state.train = state.train;
	cpu_state.truth = truth_cpu;
	cpu_state.input = in_cpu;
	forward_patch_region_layer(l, cpu_state);
	//cuda_push_array(l.output_gpu, l.output, l.batch*l.outputs);
	free(cpu_state.input);
	if (!state.train) return;
	cuda_push_array(l.delta_gpu, l.delta, l.batch*l.outputs);
	if (cpu_state.truth) free(cpu_state.truth);
}

void backward_patch_region_layer_gpu(patch_region_layer l, network_state state)
{
	flatten_ongpu(l.delta_gpu, l.h*l.w, l.classes, l.batch, 0, state.delta);
}
#endif
