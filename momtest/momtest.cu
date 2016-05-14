#include "momrun.cuh"

#define projdatastring "../momproj/rawdata/"
#define projsavestring "../momproj/saveweights/"
#define localdatastring "./"
#define localsavestring "./"
#define projsavename "momtestset3"
#define localsavename "weights"
#define projtrainstring "testset3"

//#define LOCAL

int main() {
	srand(time(NULL));
	std::string datastring;
	std::string savestring;
	std::string savename;
#ifdef LOCAL
	datastring = localdatastring;
	savestring = localsavestring;
	savename = localsavename;
#else
	datastring = projdatastring;
	savestring = projsavestring;
	savename = projsavename;
#endif

	setStrings(datastring, savestring);

	LayerCollection layers = createLayerCollection();
	initializeLayers(&layers);
	loadWeights(layers, savename);

	std::string trainstring;

#ifdef LOCAL
	std::cout << "Enter the name of an appropriate testset in the current directory: ";
	std::cin >> trainstring;
	std::cout << std::endl;
#else
	trainstring = projtrainstring;
#endif

	std::cout << "Reading trainset: ";
	size_t numDiscards[2];
	readTrainSet(trainstring, true, numDiscards);
	std::cout << "done." << std::endl;
	std::cout << numDiscards[0] << "/" << numDiscards[1] << " samples discarded." << std::endl;

	std::cout << "Running sim: " << std::endl;
	float error = testSim(layers);
	std::cout << "Error: " << error << std::endl;
	
#ifdef LOCAL
	system("pause");
#endif
}