#include "network.h"
#include "patch_region_layer.h"
#include "cost_layer.h"
#include "utils.h"
#include "parser.h"
#include "box.h"
#include "demo.h"
#include "option_list.h"

#ifdef OPENCV
#include "opencv2/highgui/highgui_c.h"
#endif

#define _CRTDBG_MAP_ALLOC  
#include <stdlib.h>  
#include <crtdbg.h>  

extern char** str_split(char* a_str, const char a_delim);

void train_patch_classifier_with_mask(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
{
	printf("Entering to read %s", datacfg);
	list *options = read_data_cfg(datacfg);
	char *train_images = option_find_str(options, "train", "data/train.list");
	char *backup_directory = option_find_str(options, "backup", "/backup/");
	
	srand(time(0));
	char *base = basecfg(cfgfile);
	printf("%s\n", base);
	float avg_loss = -1;
	network *nets = calloc(ngpus, sizeof(network));

	srand(time(0));
	int seed = rand();
	int i;
	for (i = 0; i < ngpus; ++i) {
		srand(seed);
#ifdef GPU
		cuda_set_device(gpus[i]);
#endif
		nets[i] = parse_network_cfg(cfgfile);
		if (nets[i].w % 32 != 0 || nets[i].h % 32 != 0) {
			fprintf(stderr, "dimensions should be ddivisiable by 32\n");
			system("pause");
			return;
		}
		
		if (weightfile) {
			load_weights(&nets[i], weightfile);
		}
		if (clear) *nets[i].seen = 0;
		nets[i].learning_rate *= ngpus;
	}
	srand(time(0));
	network net = nets[0];

	int imgs = net.batch * net.subdivisions * ngpus;
	printf("Learning Rate: %f, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
	data train, buffer;

	layer l = net.layers[net.n - 1];

	int classes = l.classes;

	list *plist = get_paths(train_images);
	//int N = plist->size;
	char **paths = (char **)list_to_array(plist);
	load_args args = { 0 };
	args.w = net.w;
	args.h = net.h;

	args.grid_w = l.w;
	args.grid_h = l.h;
	args.paths = paths;
	args.n = imgs;
	args.m = plist->size;
	args.classes = classes;
	args.d = &buffer;
	args.type = PATCH_CLASSIFICATION_DATA_WITH_MASK;
	args.threads = 8;

	pthread_t load_thread = load_data(args);
	clock_t time;
	int count = 0;
	//while(i*imgs < N*120){
	while (get_current_batch(net) < net.max_batches) {
		time = clock();
		pthread_join(load_thread, 0);
		train = buffer;
		load_thread = load_data(args);

		printf("Loaded: %lf seconds\n", sec(clock() - time));

		time = clock();
		float loss = 0;
#ifdef GPU
		if (ngpus == 1) {
			loss = train_network(net, train);
		}
		else {
			loss = train_networks(nets, ngpus, train, 4);
		}
#else
		loss = train_network(net, train);
#endif
		if (avg_loss < 0) avg_loss = loss;
		avg_loss = avg_loss*.9 + loss*.1;

		i = get_current_batch(net);
		printf("%d: %f, %f avg, %0.25f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), sec(clock() - time), i*imgs);
		if (i % 1000 == 0 || i<1000 && i%100==0 ) {
#ifdef GPU
			if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
			char buff[256];
			sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
			save_weights(net, buff);
		}
		free_data(train);
		_CrtDumpMemoryLeaks();
	}
#ifdef GPU
	if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
	char buff[256];
	sprintf(buff, "%s/%s_final.weights", backup_directory, base);
	save_weights(net, buff);
}

void validate_patch_classifier_with_mask(char* datacfg, char* cfgfile, char* weightfile, float iter_in_k, char* pr_rc_filename, int vis_detections, int save_detections) {
	int i, j, c, index;


	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	srand(time(0));

	list* options = read_data_cfg(datacfg);

	char* valid_list = option_find_str(options, "valid", "data/valid.list");
	int classes = option_find_int(options, "classes", 3);
	list* plist = get_paths(valid_list);

	int m = plist->size;

	layer l = net.layers[net.n - 1];

	char **paths = (char **)list_to_array(plist);


	clock_t time;
	int count = 0;
	float TP_fire = 0;
	float FP_fire = 0;
	float TP_smoke = 0;
	float FP_smoke = 0;

	float FN_fire = 0;
	float FN_smoke = 0;

	float TN = 0;

	const int background_cls_id = l.classes - 1;
	const int fire_cls_id = 0;
	const int smoke_cls_id = 1;

	int num_background_cls = 0;
	FILE * file = fopen(pr_rc_filename, "r");

	//put a header to log file if it is opening for the first time
	if (file) {
		fclose(file);
		file = fopen(pr_rc_filename, "a+");
	}
	else {
		file = fopen(pr_rc_filename, "w");
		printf(file, "iter_in_k \t FirePrecision \t FireRecall \t SmokePrecision \t SmokeRecall\n");
	}

	while (count < m) {
		time = clock();
		printf("processing paths[%d] = %s\n", count, paths[count]);


		//image im = load_image_color(paths[count], 0, 0);
		char* path = calloc(strlen(paths[count]), sizeof(char));
		strncpy(path, paths[count], strlen(paths[count]));

		char** tokens = str_split(path, ',');
		if (!tokens)
		{
			printf("Error occured in reading images\n");
			exit(0);
		}
		if (!*(tokens + 0) || !*(tokens + 1)) {
			printf("Error in splitting %c\n", path);
			exit(0);
		}
		image im = load_image_color_with_mask(*(tokens + 0), *(tokens + 1), 0, 0);			
		image sized = resize_image(im, net.w, net.h);
		layer l = net.layers[net.n - 1];

		float *X = sized.data;
		float *predictions = network_predict(net, X);

		float *truths = calloc(l.w*l.h*l.classes, sizeof(float));
		fill_grid_truth(*(tokens +0), truths, l.classes, l.w, l.h);

		free(*(tokens + 0));
		free(*(tokens + 1));
		free(tokens);
		free(path);
		
		for (j = 0; j < l.h; j++) {
			for (i = 0; i < l.w; i++) {
				index = j*l.w*l.classes + i*l.classes;
				if (truths[index + background_cls_id] == -1) continue; //it means this is an ambiguous cell				
				int cls_id = max_index(truths + index, l.classes);
				int prediction = max_index(predictions + index, l.classes);

				if (prediction == background_cls_id) {
					if (prediction == cls_id) TN++;
					else {
						if (cls_id == fire_cls_id) FN_fire++;
						else if (cls_id == smoke_cls_id) FN_smoke++;
						else {
							printf("false negative has to be either fire or smoke\n");
							system("pause");
							return 0;
						}
					}
				}
				else if (prediction == fire_cls_id) {
					if (prediction == cls_id) TP_fire++;
					else FP_fire++;
				}
				else if (prediction == smoke_cls_id) {
					if (prediction == cls_id)TP_smoke++;
					else FP_smoke++;
				}
			}
		}
		if (vis_detections) {
			draw_patch_detections(sized, predictions, l.w, l.h, l.classes);
			show_image(sized, "predictions");
			cvWaitKey(40);
		}
		if (save_detections) {
			draw_patch_detections(sized, predictions, l.w, l.h, l.classes);
			char str[1024];
			find_replace(paths[count], "images", "detect_images", str);
			save_image(sized, str);
		}
		//free(predictions);
		free(truths);
		free_image(im);
		free_image(sized);
		count++;
	}

	fprintf(file, "%0.1f:  %f   %f   %f   %f\n", iter_in_k, TP_fire / (TP_fire + FP_fire), TP_fire / (TP_fire + FN_fire), TP_smoke / (TP_smoke + FP_smoke), TP_smoke / (TP_smoke + FN_smoke));
}

void validate_patch_classifier_with_mask_globally(char* datacfg, char* cfgfile, char* weightfile, float iter_in_k, char* pr_rc_filename, int vis_detections, int save_detections) {
	int i, j, c, index;	 
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);
	srand(time(0));

	list* options = read_data_cfg(datacfg);

	char* valid_list = option_find_str(options, "valid", "data/valid.list");
	int classes = option_find_int(options, "classes", 3);
	list* plist = get_paths(valid_list);

	int m = plist->size;
	
	layer l = net.layers[net.n - 1];

	char **paths = (char **)list_to_array(plist);

	clock_t time;
	int count = 0;
	float TP_fire = 0;
	float FP_fire = 0;
	float TP_smoke = 0;
	float FP_smoke = 0;

	float FN_fire = 0;
	float FN_smoke = 0;

	float TN = 0;

	const int background_cls_id = l.classes-1;
	const int fire_cls_id = 0;
	const int smoke_cls_id = 1;

	int num_background_cls = 0;
	FILE * file = fopen(pr_rc_filename, "r");
	
	//put a header to log file if it is opening for the first time
	if (file) {
		fclose(file);
		file = fopen(pr_rc_filename, "a+");
	}
	else {
		file = fopen(pr_rc_filename, "w");
		printf(file, "iter_in_k \t FirePrecision \t FireRecall \t SmokePrecision \t SmokeRecall\n");
	}
	int isFire = 0;
	int firePredicted = 0;
	int isSmoke = 0;
	int smokePredicted = 0;

	while (count < m) {
		time = clock();
		printf("processing paths[%d] = %s\n", count, paths[count]);

		char* path = calloc(strlen(paths[count]), sizeof(char));
		strncpy(path, paths[count], strlen(paths[count]));

		char** tokens = str_split(path, ',');
		if (!tokens)
		{
			printf("Error occured in reading images\n");
			exit(0);
		}
		if (!*(tokens + 0) || !*(tokens + 1)) {
			printf("Cannot split the line %c",path);
			exit(0);
		}
		image im = load_image_color_with_mask(*(tokens + 0), *(tokens + 1), 0, 0);
		
		image sized = resize_image(im, net.w, net.h);
		layer l = net.layers[net.n - 1];

		float *X = sized.data;
		float *predictions = network_predict(net, X);

		float *truths = calloc(l.w*l.h*l.classes, sizeof(float));
		fill_grid_truth(*(tokens + 0), truths, l.classes, l.w, l.h);

		free(*(tokens + 0));
		free(*(tokens + 1));
		
		free(tokens);
		free(path);
		
		int positive = 0;
		isFire = 0;
		firePredicted = 0;
		isSmoke = 0;
		smokePredicted = 0;

		for (j = 0; j < l.h; j++) {
			for (i = 0; i < l.w; i++) {
				
				index = j*l.w*l.classes + i*l.classes;				
				if (truths[index + background_cls_id] == -1) continue; //it means this ambiguous cell				
				int cls_id = max_index(truths + index, l.classes);
				
				if (cls_id == fire_cls_id) { 
					isFire = 1;
				}
				else if (cls_id == smoke_cls_id) {
					isSmoke = 1;
				}

				int prediction = max_index(predictions + index, l.classes);
				if (prediction == fire_cls_id) {
					firePredicted = 1;
				}
				else if (prediction == smoke_cls_id) {
					smokePredicted = 1;
				}
			}
		}
		if (isFire)
			if (firePredicted)
				TP_fire++;
			else
				FN_fire++;
		else
			if (firePredicted)
				FP_fire++;

		
		if (isSmoke)
			if (smokePredicted)
				TP_smoke++;
			else
				FN_smoke++;
		else
			if (smokePredicted)
				FP_smoke++;

	
		if (vis_detections) {
			draw_patch_detections(sized, predictions, l.w, l.h, l.classes);
			show_image(sized, "predictions");
			cvWaitKey(40);
		}
		if (save_detections) {
			draw_patch_detections(sized, predictions, l.w, l.h, l.classes);
			char str[1024] ;
			find_replace(paths[count], "images", "detect_images", str);
			save_image(sized, str);
		}
		//free(predictions);
		free(truths);
		free_image(im);
		free_image(sized);
		count++;
	}

	fprintf(file, "%0.1f:  %f   %f   %f   %f\n", iter_in_k, TP_fire / (TP_fire + FP_fire), TP_fire/ (TP_fire+FN_fire), TP_smoke / (TP_smoke + FP_smoke), TP_smoke / (TP_smoke + FN_smoke));
	fclose(file);
}

void test_patch_classifier_with_mask(char* datacfg, char *cfgfile, char* weightfile, char *filename) {
	network net = parse_network_cfg(cfgfile);
	if (weightfile) {
		load_weights(&net, weightfile);
	}
	set_batch_network(&net, 1);

	clock_t time;

	image im = load_image_color(filename, 0, 0);
	image sized = resize_image(im, net.w, net.h);
	layer l = net.layers[net.n - 1];

	float *X = sized.data;
	float *predictions = network_predict(net, X);

	int i, j,c;
	
	for (j = 0; j < l.h; j++){
		for (i = 0; i < l.w; i++) {
			printf("(%d,%d) (", j, i );
			for (c = 0; c < l.classes - 1; c++) {
				int index = j*l.w*l.classes + i*l.classes + c;
				printf("%f,", predictions[index]);
			}
			int index = j*l.w*l.classes + i*l.classes + (l.classes-1);
			printf("%f)\n", predictions[index]);
		}
	}

	draw_patch_detections(sized, predictions, l.w, l.h, l.classes);
	show_image(sized, "predictions");
	cvWaitKey(0);

	free_image(im);
	free_image(sized);

	cvDestroyAllWindows();
	system("pause");
	
}

void run_patch_classifier_with_mask(int argc, char **argv)
{
	char *prefix = find_char_arg(argc, argv, "-prefix", 0);
	float thresh = find_float_arg(argc, argv, "-thresh", .24);
	int cam_index = find_int_arg(argc, argv, "-c", 0);
	int frame_skip = find_int_arg(argc, argv, "-s", 0);
	float iter = find_float_arg(argc, argv, "-iter", 1000);
	float iter_in_k = iter / 1000.;

	int vis_detections = find_int_arg(argc, argv, "-v_d", 0);
	int save_detections = find_int_arg(argc, argv, "-s_d", 0);

	float *pr_rc_filename = find_char_arg(argc, argv, "-pr_r_file", "log/pr_r_file.txt");
	if (argc < 4) {
		fprintf(stderr, "usage: %s %s [train/test/valid] [cfg] [weights (optional)]\n", argv[0], argv[1]);
		return;
	}
	char *gpu_list = find_char_arg(argc, argv, "-gpus", 0);
	int *gpus = 0;
	int gpu = 0;
	int ngpus = 0;
	if (gpu_list) {
		printf("%s\n", gpu_list);
		int len = strlen(gpu_list);
		ngpus = 1;
		int i;
		for (i = 0; i < len; ++i) {
			if (gpu_list[i] == ',') ++ngpus;
		}
		gpus = calloc(ngpus, sizeof(int));
		for (i = 0; i < ngpus; ++i) {
			gpus[i] = atoi(gpu_list);
			gpu_list = strchr(gpu_list, ',') + 1;
		}
	}
	else {
		gpu = gpu_index;
		gpus = &gpu;
		ngpus = 1;
	}

	int clear = find_arg(argc, argv, "-clear");

	char *datacfg = argv[3];
	char *cfg = argv[4];
	char *weights = (argc > 5) ? argv[5] : 0;
	char *filename = (argc > 6) ? argv[6] : 0;
	if (0 == strcmp(argv[2], "test")) test_patch_classifier_with_mask(datacfg, cfg, weights, filename);
	else if (0 == strcmp(argv[2], "train")) train_patch_classifier_with_mask(datacfg, cfg, weights, gpus, ngpus, clear);
	else if (0 == strcmp(argv[2], "valid")) validate_patch_classifier_with_mask(datacfg, cfg, weights,iter_in_k,pr_rc_filename,vis_detections,save_detections);
	else if (0 == strcmp(argv[2], "valid_global")) validate_patch_classifier_with_mask_globally(datacfg, cfg, weights, iter_in_k, pr_rc_filename, vis_detections, save_detections);	
	else if (0 == strcmp(argv[2], "demo_mask")) {
		list *options = read_data_cfg(datacfg);
		int classes = option_find_int(options, "classes", 20);
		//demo_patch_classifier_with_mask(cfg, weights, thresh, cam_index, filename, classes, frame_skip, prefix);
	}
}
