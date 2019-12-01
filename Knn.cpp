#include <iostream>
#include <ctime>    // For time()
#include <cstdlib>  // For srand() and rand()
#include <math.h>
#include <unordered_map>
using namespace std;


void show_usage( const char* prog )
{
   printf("Usage for %s\n", prog);
   printf("\t-feat     <int value> : # of features. (100)\n");
   printf("\t-label    <int value> : # of labels. (2)\n");
   printf("\t-count    <int value> : # of train samples. (1000)\n");
   printf("\t-test     <int value> : # of test samples.   (200)\n");
   printf("\t-k        <int value> : K value.   (5)\n");
}

int main(int argc, char * argv[]) {
	srand(time(0));

	int features = 100;
	int labels = 2;
	int sample_count = 1000;
	int test_count = 200;
	int k = 5;

	for (int i = 1; i < argc; i++)
	{
		std::string arg = argv[i];
		if (arg == "-feat")
		{
			if ((i+1) >= argc) { fprintf(stderr,"Missing value for -feat\n"); show_usage(argv[0]); return 1; }
			features = atoi( argv[i+1] );  //atoi() takes str and converts to int
			i++;
		}
		else if (arg == "-label")
		{
			if ((i+1) >= argc) { fprintf(stderr,"Missing value for -label\n"); show_usage(argv[0]); return 1; }
			labels = atoi( argv[i+1] );
			i++;
		}
		else if (arg == "-count")
		{
			if ((i+1) >= argc) { fprintf(stderr,"Missing value for -count\n"); show_usage(argv[0]); return 1; }
			sample_count = atoi( argv[i+1] );
			i++;
		}
		else if (arg == "-test")
		{
			if ((i+1) >= argc) { fprintf(stderr,"Missing value for -test\n"); show_usage(argv[0]); return 1; }
			test_count = atoi( argv[i+1] );
			i++;
		}
		else if (arg == "-k")
		{
			if ((i+1) >= argc) { fprintf(stderr,"Missing value for -k\n"); show_usage(argv[0]); return 1; }
			k = atoi( argv[i+1] );
			i++;
		}
		else if (arg == "--help" || arg == "-h")
		{
			show_usage(argv[0]); return 0;
		}
		else
		{
			fprintf(stderr,"Unknown option %s\n", arg.c_str());
			show_usage(argv[0]);
			return 1;
		}
	}

	printf("features: %d\n", features);
	printf("labels: %d\n", labels);
	printf("count: %d\n", sample_count);
	printf("test: %d\n", test_count);
	printf("k: %d\n", k);

	//Creating training set and fill the set with random numbers;
	double *samples = new double[sample_count*(features+1)];
	for(int i=0; i<sample_count; i++) {
	 	for(int j=0; j<features; j++) {
	 	 	samples[i*(features+1)+j] = rand()/(RAND_MAX+1.);
	 	}
	 	samples[i*(features+1)+features] = rand()%labels;
	}

	//Creating test set and fill the set with random numbers.
	double *testset = new double[test_count*features];
	for(int i=0; i<test_count; i++) {
		for(int j=0; j<features; j++) {
			testset[i*features+j] = rand()/(RAND_MAX+1.);
		}
	}

	int *prediction = new int[test_count];
	//Going through test set
	for(int t=0; t<test_count; t++) {
	 	//array that will save nearest neighbours.
	 	//each element will save nearest neighbours distance and its label
	 	double (*nearest_neighbours)[2] = new double[k][2];

	 	//going through training set.
	 	for(int s=0; s<sample_count; s++) {

	 		//distance will be calculated by euclidean distance. There is no need to take square root.
	 		double dist=0;

	 		//going through features.
	 		for(int f=0; f<features; f++) {
        	 	 dist += pow(samples[s*(features+1)+f]-testset[t*features+f],2);
        	}

			/*
			check this sample is nearest neighbour.
			nearest_neighbours are placed farthest to nearest in the array.
			or maybe they might be empty meaning distance equal to 0
			*/
			int n = 0;

			/*
			finding place this sample to be placed in nearest neighbours.
			*/
			for(n=0; n<k; n++) {
				if(nearest_neighbours[n][0]==0||dist<nearest_neighbours[n][0]) continue;
			}

			/*
			if this sample should be placed in nearest neighbours,
			farther neighbours relative to this samples should be shifted by one place.
			*/
			for(int m=1; m<n; m++) {
				nearest_neighbours[m-1][0]=nearest_neighbours[m][0];
				nearest_neighbours[m-1][1]=nearest_neighbours[m][1];
			}

			/*
			placing this sample in the nearest neighbrours if it should be.
			*/
			if(n>0) {
				nearest_neighbours[n-1][0]=dist;
				nearest_neighbours[n-1][1]=samples[s*(features+1)+features];
			}
	 	}

	 	/*
	 	finding most frequent label.
	 	first create frequency mapping.
	 	*/
	 	unordered_map<int, int> mp;
		for(int n=0; n<k; n++) {
            mp[(int)nearest_neighbours[n][1]]++;
		}

		/*
 		going through frequency mapping to find most frequent one.
		*/
		int max_count = 0, res = -1;
		for (auto n : mp) {
			if (max_count < n.second) {
				res = n.first;
				max_count = n.second;
			}
		}

		prediction[t] = res;
	}

	for(int t=0; t<test_count; t++) {
	 	printf("%d, ", prediction[t]);
	}
}
