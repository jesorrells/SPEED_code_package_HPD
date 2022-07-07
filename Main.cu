//Program for performing GPU-accelerated single-photon peak event detection for FLIM with option for multiple
//thresholds, such as when using a hybrid photodetector (HPD)
//By Janet Sorrells and Rishee Iyer, 2022
//Contact: janetes2@illinois.edu
//See: https://doi.org/10.1021/acsphotonics.2c00505 (SPEED with HPD, 2022)
// and https://doi.org/10.1364/OE.439675 (SPEED with PMT, 2021)

/*
Description of variables from Config.txt file for you to provide:
numX - number of pixels along the fast axis  (points per line)
numY - number of pixels along the slow axis  (lines per frame)
num2T - number of timepoints sampled per pixel per two laser pulses
numTtot - number of timepoints sampled per pixel per frame
	(ex: if for each pixel you sample at 5.0 GHz, using laser with rep rate 80 MHz, 
	and have a pixel dwell time of 5 us = 400 laser pulses, numTtot = 25000, num2T = 125)
nChunks - number of partitions within the image, this is useful for real-time processing so 
	we can transfer data in smaller "chunks" to the PC. You can set it to 1 for post-processing, 
	generally for a 512x512x25000 16-bit dataset, we use nChunks = 32. 
numF - number of frames
Thresh - Intensity threshold (in # of photons) to determine the minimum intensity for lifetime calculations
FileBase - name of the file to open
OutputExt - string to add as an extension to the output data file names

*/

#include "DLLFunctions.cuh"

int main(int argc, char* argv[])
{
	if (argc > 1)
	{
		FILE* ConfigFile = fopen(argv[1], "rt");

		int32_t numX, numY, num2T, numTtot, nChunks, numF, numYPerChunk;

		float Thresh;
		char FileBase[1024], OutputExt[1024];

		fscanf(ConfigFile, "%s", &FileBase[0]);
		fscanf(ConfigFile, "%d", &numX);
		fscanf(ConfigFile, "%d", &numY);
		fscanf(ConfigFile, "%d", &num2T);
		fscanf(ConfigFile, "%d", &numTtot);
		fscanf(ConfigFile, "%d", &nChunks);
		fscanf(ConfigFile, "%d", &numF);
		fscanf(ConfigFile, "%f", &Thresh);
		fscanf(ConfigFile, "%s", &OutputExt[0]);

		fclose(ConfigFile);

		printf("----------------------------------------------------\n    FAST FLIM POST PROCESSING LOG     \n----------------------------------------------------\n");

		char DataFileName[1024], OutputMPMFile[1024], OutputFLIMFile[1024], OutputGFile[1024], OutputSFile[1024], OutputLogFile[1024];
		strcpy(DataFileName, FileBase);
		strcat(DataFileName, "_FLIMRaw.bin");

		strcpy(OutputMPMFile, FileBase);
		strcat(OutputMPMFile, OutputExt);
		strcat(OutputMPMFile, "_MPMOut.bin");

		strcpy(OutputFLIMFile, FileBase);
		strcat(OutputFLIMFile, OutputExt);
		strcat(OutputFLIMFile, "_FLIMOut.bin");

		strcpy(OutputGFile, FileBase);
		strcat(OutputGFile, OutputExt);
		strcat(OutputGFile, "_GOut.bin");

		strcpy(OutputSFile, FileBase);
		strcat(OutputSFile, OutputExt);
		strcat(OutputSFile, "_SOut.bin");

		strcpy(OutputLogFile, FileBase);
		strcat(OutputLogFile, OutputExt);
		strcat(OutputLogFile, "_LogFile.txt");

		FILE* FIDLog = fopen(OutputLogFile, "wt");

		numYPerChunk = numY / nChunks;

		printf("Raw data file:    %s\n", DataFileName);
		printf("Output MPM file:  %s\n", OutputMPMFile);
		printf("Output FLIM file: %s\n", OutputFLIMFile);
		printf("Output G values:  %s\n", OutputGFile);
		printf("Output S values:  %s\n", OutputSFile);
		printf("Output Log:       %s\n", OutputLogFile);
		printf("%d x %d x %d x %d\n", numTtot, numX, numY, numF);
		printf("%d points per two pulses | %d chunks per frames\n", num2T, nChunks);
		printf("Threshold: %1.2f\n", Thresh);

		fprintf(FIDLog, "Raw data file:    %s\n", DataFileName);
		fprintf(FIDLog, "Output MPM file:  %s\n", OutputMPMFile);
		fprintf(FIDLog, "Output FLIM file: %s\n", OutputFLIMFile);
		fprintf(FIDLog, "Output G values:  %s\n", OutputGFile);
		fprintf(FIDLog, "Output S values:  %s\n", OutputSFile);
		fprintf(FIDLog, "%d x %d x %d x %d\n", numTtot, numX, numY, numF);
		fprintf(FIDLog, "%d points per two pulses | %d chunks per frames\n", num2T, nChunks);
		fprintf(FIDLog, "Threshold: %1.2f\n", Thresh);


		//Initialize and start GPU
		cudaError_t Ret0 = FastFLIMGPU_StartGPU();
		cudaError_t Ret1 = FastFLIMGPU_InitializeAndAllocateMemory(&numX, &numY, &numTtot, &num2T, &nChunks);
		printf("Initialized (%s)\n", cudaGetErrorString(Ret1));


		//Allocate memory
		int16_t* Input = (int16_t*)malloc(sizeof(int16_t) * numX * numY * numTtot);
		float* MPMImage_NoAv = (float*)malloc(sizeof(float) * numX * numY);
		float* OutputMPM = (float*)malloc(sizeof(float) * numX * numY);
		float* OutputFLIM = (float*)malloc(sizeof(float) * numX * numY);
		uint16_t* OutputS = (uint16_t*)malloc(sizeof(uint16_t) * numX * numY);
		uint16_t* OutputG = (uint16_t*)malloc(sizeof(uint16_t) * numX * numY);
		float* OutputHistogram = (float*)malloc(sizeof(float) * numX * numY * num2T);

		//Initialize file index and chunk index to 0
		int32_t fidx = 0, chidx, lidx = 0;;

		//Open output files
		FILE* FIDData = fopen(DataFileName, "rb");
		FILE* FIDMPM = fopen(OutputMPMFile, "wb");
		FILE* FIDFLIM = fopen(OutputFLIMFile, "wb");
		FILE* FIDG = fopen(OutputGFile, "wb");
		FILE* FIDS = fopen(OutputSFile, "wb");

		clock_t t = clock(), FTrack = 0, FBeg, FEnd, FOverall;
		for (fidx = 0; fidx < numF; fidx++) //loop through frames
		{
			for (lidx = 0; lidx < numY; lidx++) //loop through chunks
			{
				fread(Input, sizeof(int16_t), numX * numTtot, FIDData); //read each chunk
				FBeg = clock();
				//Perform peak detection on raw data and average from numX x numYPerChunk x numTtot to numX x numYPerChunk x numT
				cudaError_t Ret2 = FastFLIMGPU_ReportFrame(Input, numX, numY, numTtot, num2T, nChunks, lidx);
				FEnd = clock() - FBeg;
				FTrack += FEnd;
			}

			
			//Once all chunks of a single frame are ready, process the whole frame

			FBeg = clock();

			//Perform shifting, spatial binning, and lifetime calculations on whole frame
			cudaError_t Ret2 = FastFLIMGPU_DoFastFLIM(OutputMPM, OutputHistogram, OutputFLIM, OutputG, OutputS,
				numX, numY, num2T, &Thresh, 0, "_", 0, "_");

			FEnd = clock() - FBeg;
			FTrack += FEnd;

			//Write output files
			fwrite(OutputFLIM, sizeof(float), numX * numY, FIDFLIM); //mean fluorescence lifetime image, numX x numY
			fwrite(OutputMPM, sizeof(float), numX * numY, FIDMPM); //intensity, numX x numY
			fwrite(OutputHistogram, sizeof(float), numX * numY * num2T, FIDMPM); //OutputHistogram shows the decay for each pixel, numX x numY x num2T
			fwrite(OutputG, sizeof(uint16_t), numX * numY, FIDG); //g from phasor analysis, numX x numY
			fwrite(OutputS, sizeof(uint16_t), numX * numY, FIDS); //s from phasor analysis, numX x numY
			
			FOverall = clock() - t;
			printf("\rFinished %d/%d frames in %1.3f seconds...                        ", fidx + 1, numF, ((float)FOverall) / CLOCKS_PER_SEC);
		}

		fprintf(FIDLog, "\nJust processing took %1.3f seconds.\n", ((float)FTrack) / CLOCKS_PER_SEC);
		fprintf(FIDLog, "The whole shebang took %1.3f seconds.\n", ((float)FOverall) / CLOCKS_PER_SEC);

		printf("\nJust processing took %1.3f seconds...\n", ((float)FTrack) / CLOCKS_PER_SEC);

		cudaError_t Ret3 = FastFLIMGPU_DestroyEverything();
		printf("\nDestroyed everything (%s)\n", cudaGetErrorString(Ret3));
		fclose(FIDData);
		fclose(FIDFLIM);
		fclose(FIDMPM);
		fclose(FIDS);
		fclose(FIDG);

		free(Input);
		free(MPMImage_NoAv);
		free(OutputMPM);
		free(OutputFLIM);
		free(OutputG);
		free(OutputS);
		free(OutputHistogram);
	}
	else
	{
	printf("YOU DIDN'T SEND ANY ARGUMENTS!");
	}
	return 0;
}