#include "momrun.cuh"
#include "Shlwapi.h"

#define nIter 5
#define discardSamples true

#define datastring "rawdata/"
#define savestring "saveweights/"
#define savename "train"
#define trainstring "trainset"

void saveResults(size_t numRuns, float beforeError, float afterError);
void loadParameters();
void saveParameters();

size_t numRuns;

float stepMultiplier(size_t numRuns) {
	return 1.0f;
}

int main() {
	srand(time(NULL));
	setStrings(datastring, savestring);
	numRuns = 0;

	loadParameters();

	LayerCollection layers = createLayerCollection();
	initializeLayers(&layers);
	loadWeights(layers, savename);

	while (true) {
		std::cout << "Reading trainset: ";
		auto readstart = std::chrono::high_resolution_clock::now();
		size_t numDiscards[2];
		readTrainSet(trainstring, discardSamples, numDiscards);
		auto readelapsed = std::chrono::high_resolution_clock::now() - readstart;
		long long readtime = std::chrono::duration_cast<std::chrono::microseconds>(readelapsed).count();
		std::cout << readtime / 1000000 << " s" << std::endl;
		std::cout << numDiscards[0] << "/" << numDiscards[1] << " samples discarded." << std::endl;

		std::cout << nIter << "+2 runs: ";
		auto gpustart = std::chrono::high_resolution_clock::now();

		float beforeError = runSim(layers, false, 0);

		for (size_t i = 0; i < nIter; i++) {
			runSim(layers, true, stepMultiplier(numRuns));
		}

		//runSim(layers, true, 0.01f);
		float afterError = runSim(layers, false, 0);

		auto gpuelapsed = std::chrono::high_resolution_clock::now() - gpustart;
		long long gputime = std::chrono::duration_cast<std::chrono::microseconds>(gpuelapsed).count();
		saveWeights(layers, savename);
		numRuns += nIter;
		std::cout << gputime/1000000 << " s, Errors: before - " << beforeError << ", after - " << afterError << std::endl;

		saveResults(numRuns, beforeError, afterError);
		saveParameters();
	}
}

void saveResults(size_t numRuns, float beforeError, float afterError) {
	std::stringstream resname;
	resname << savestring << savename << "result";
	std::ofstream resfile(resname.str().c_str(), std::ios_base::app);
	resfile << numRuns << " " << beforeError << " " << afterError << std::endl;
}

void loadParameters() {
	std::stringstream pss;
	pss << savestring << savename << "pars";
	std::ifstream infile(pss.str().c_str());

	std::string line;
	while (getline(infile, line)) {
		std::stringstream lss(line);
		std::string var;
		lss >> var;
		if (var == "numRuns")
			lss >> numRuns;
	}
}

void saveParameters() {
	std::stringstream pss;
	pss << savestring << savename << "pars";
	std::ofstream outfile(pss.str().c_str());

	outfile << "numRuns " << numRuns;
}
