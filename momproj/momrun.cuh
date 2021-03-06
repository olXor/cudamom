#ifndef MOMRUN_HEADER
#define MOMRUN_HEADER

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <helper_cuda.h>
#include "params.h"
#include "kernel.cuh"
#include <chrono>
#include <iostream>
#include <sstream>
#include <vector>
#include <list>
#include <memory>
#include <Shlwapi.h>

struct IOPair {
	std::vector<float> inputs;
	float correctoutput;
	size_t samplenum;
};

struct LayerCollection {
	std::vector<ConvolutionMatrices> convMat;
	std::vector<ConvolutionParameters> convPars;
	std::vector<MaxPoolMatrices> mpMat;
	std::vector<MaxPoolParameters> mpPars;
	std::vector<FixedNetMatrices> fixedMat;
	std::vector<FixedNetParameters> fixedPars;

	std::vector<ConvolutionMatrices*> d_convMat;
	std::vector<ConvolutionParameters*> d_convPars;
	std::vector<MaxPoolMatrices*> d_mpMat;
	std::vector<MaxPoolParameters*> d_mpPars;
	std::vector<FixedNetMatrices*> d_fixedMat;
	std::vector<FixedNetParameters*> d_fixedPars;

	size_t numConvolutions;
	size_t numFixedNets;
};

void setStrings(std::string data, std::string save);
void freeMemory();
void readTrainSet(std::string learnsetname, bool discard = false, size_t* numDiscards = NULL);
float runSim(LayerCollection layers, bool train, float customStepFactor, bool print = false);
float testSim(LayerCollection layers, std::string ofname = "");
void saveWeights(LayerCollection layers, std::string fname);
void loadWeights(LayerCollection layers, std::string fname);

void initializeLayers(LayerCollection* layers);
void initializeConvolutionMatrices(ConvolutionMatrices* mat, ConvolutionParameters* pars);
void initializeMPMatrices(MaxPoolMatrices* mat, MaxPoolParameters* pars);
void initializeFixedMatrices(FixedNetMatrices* mat, FixedNetParameters* pars, bool last);
void copyLayersToDevice(LayerCollection* layers);

void calculate(LayerCollection layers);
void backPropagate(LayerCollection layers);

LayerCollection createLayerCollection();

#endif