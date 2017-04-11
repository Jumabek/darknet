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

void train_patch_classifier(char *datacfg, char *cfgfile, char *weightfile, int *gpus, int ngpus, int clear)
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
	printf("Learning Rate: %g, Momentum: %g, Decay: %g\n", net.learning_rate, net.momentum, net.decay);
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
	args.type = PATCH_CLASSIFICATION_DATA;
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

		/*
		int k;
		for(k = 0; k < l.max_boxes; ++k){
		box b = float_to_box(train.y.vals[10] + 1 + k*5);
		if(!b.x) break;
		printf("loaded: %f %f %f %f\n", b.x, b.y, b.w, b.h);
		}
		image im = float_to_image(448, 448, 3, train.X.vals[10]);
		int k;
		for(k = 0; k < l.max_boxes; ++k){
		box b = float_to_box(train.y.vals[10] + 1 + k*5);
		printf("%d %d %d %d\n", truth.x, truth.y, truth.w, truth.h);
		draw_bbox(im, b, 8, 1,0,0);
		}
		save_image(im, "truth11");
		*/

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
		printf("%d: %f, %f avg, %f rate, %lf seconds, %d images\n", get_current_batch(net), loss, avg_loss, get_current_rate(net), sec(clock() - time), i*imgs);
		if (i % 1000 == 0 || (i < 1000 && i % 100 == 0)) {
#ifdef GPU
			if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
			char buff[256];
			sprintf(buff, "%s/%s_%d.weights", backup_directory, base, i);
			save_weights(net, buff);
		}
		free_data(train);
	}
#ifdef GPU
	if (ngpus != 1) sync_nets(nets, ngpus, 0);
#endif
	char buff[256];
	sprintf(buff, "%s/%s_final.weights", backup_directory, base);
	save_weights(net, buff);
}



void test_patch_classifier(char* datacfg, char *cfgfile, char* weightfile, char *filename) {
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
	cvDestroyAllWindows();
	system("pause");
	
}

void run_patch_classifier(int argc, char **argv)
{
	char *prefix = find_char_arg(argc, argv, "-prefix", 0);
	float thresh = find_float_arg(argc, argv, "-thresh", .24);
	int cam_index = find_int_arg(argc, argv, "-c", 0);
	int frame_skip = find_int_arg(argc, argv, "-s", 0);
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
	if (0 == strcmp(argv[2], "test")) test_patch_classifier(datacfg, cfg, weights, filename);
	else if (0 == strcmp(argv[2], "train")) train_patch_classifier(datacfg, cfg, weights, gpus, ngpus, clear);
	else if (0 == strcmp(argv[2], "valid")) validate_detector(datacfg, cfg, weights);
	else if (0 == strcmp(argv[2], "recall")) validate_detector_recall(cfg, weights);
	else if (0 == strcmp(argv[2], "demo")) {
		list *options = read_data_cfg(datacfg);
		int classes = option_find_int(options, "classes", 20);
		char *name_list = option_find_str(options, "names", "data/names.list");
		char **names = get_labels(name_list);
		demo_patch_classifier(cfg, weights, thresh, cam_index, filename, names, classes, frame_skip, prefix);
	}
}
