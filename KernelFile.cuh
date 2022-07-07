//Program for performing GPU-accelerated single-photon peak event detection for FLIM with option for multiple
//thresholds, such as when using a hybrid photodetector (HPD)
//By Janet Sorrells and Rishee Iyer, 2022
//Contact: janetes2@illinois.edu
//See: https://doi.org/10.1021/acsphotonics.2c00505 (SPEED with HPD, 2022)
// and https://doi.org/10.1364/OE.439675 (SPEED with PMT, 2021)

//Include header files
#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <windows.h>
#include <stdio.h>
#include <conio.h>
#include <iostream>
#include <stddef.h>
#include <memoryapi.h>
#include <stdlib.h>
#include <stdint.h>
#include <cuda.h>
#include <cufft.h>
#include <cublas_v2.h>
#include <cassert>
#include <time.h>

//Define fixed variables
#define DLLEXPORT extern "C" __declspec(dllexport)
#define BLOCK_SIZE 16
#define BLOCK_SIZE_4 4
#define BLOCK_SIZE_T 5
#define FILESTUFFEH 1
#define BINSTEP 1 //BINSTEP is the number of adjacent pixels used for imaging, BINSTEP = 1 --> 3x3 binning
#define PI 3.14159265f
#define DECAYLENGTH 125 
//see paper to determine how to set HPD thresholds, it will depend on the bias voltage, amplifier, digitizer, etc.
#define PEAKTHRESH_HPD_1P 350  
#define PEAKTHRESH_HPD_2P 2150  
#define PEAKTHRESH_HPD_3P 4180 
#define PEAKTHRESH_HPD_4P 5850 
#define PEAKTHRESH_HPD_5P 7150 
#define TAXIS_COEFF 2.0f * PI * 80E6f / (5.0E9f)
#define LT_COEFF (1E12f/(2.0f*PI*80E6f))
#define FLIMLOGFILENAME "C:\\FLIM Acquisiton Software GPU Version\\FLIMProcLogFile_v109.txt" //Set this where you want the log file to go

typedef uint8_t CountDataType; 
typedef uint16_t AvgDataType;

//GPU pointers and variables
/* ALL THE GPU POINTERS AND VARIABLES. I MEAN, ALL OF THEM HAVE A PREFIX GPU or G. GOD, DUH! */
extern int16_t* GPU_FLIMRawData_1Line;
extern float* GPU_numXxnumY_MeanLifetime_AfterAv, * GPU_numXxnumY_MPMImage_BeforeAv;
extern uint16_t* GPU_numXxnumY_S_AfterAv, * GPU_numXxnumY_G_AfterAv;
extern uint8_t* GPU_Thresh_Array;
extern float* gThresh;
extern int32_t* gnumX, * gnumY, * gnumTTot, * gnum2T, * gDivFactor, * gNumChunks, * gLineIndex, * gnumYPerChunk, * gnumXnumYnumT, * gNumxidMax;
extern cudaStream_t FastFLIMStream, OCMStream;
extern int CudaLeastPriority, CudaGreatestPriority;
extern CountDataType* GPU_numXxnumTxnumPulses_Count_Container;
extern AvgDataType* GPU_numXxnumYxnumT_Container, * GPU_numXxnumYxnumT_Container_Shifted;

/*GPU Kernels below!!*/

/* Average counts to a "pulse" (create a histogram of photon counts vs. time where the time corresponds to the laser period)
InData: photon counts numX x numYPerChunk x numTtot
OutData: photon counts numX x numYPerChunk x numT 
*/
__global__ void AverageCountsToAPulse(CountDataType* InData, AvgDataType* OutData, int32_t* numX, int32_t* numTTot,
	int32_t* numT, int32_t* DivFactor, int32_t* LineIndex)
{
	long xid = blockIdx.x * blockDim.x + threadIdx.x;
	long tid = blockIdx.z * blockDim.z + threadIdx.z;
	size_t tidx;
	size_t inidx = (size_t)tid + (size_t)xid * (size_t)*numTTot;

	AvgDataType tempOut = 0;
	for (tidx = 0; tidx < *DivFactor; tidx++)
	{
		tempOut += (AvgDataType)InData[inidx + tidx * (size_t)*numT];
	}
	OutData[(size_t)tid + (size_t)xid * (size_t)*numT + (size_t)*numX * (size_t)*numT * (size_t)*LineIndex] = tempOut;

}

/*Count photons - Single-photon PEak Event Detection (SPEED)!!!!
InData: PMT output amplified and digitized (12-bit), numX x numYperChunk x numTtot
OutData: 0 for no photon count, 1 for photon count, 2 for 2 photon counts... numX x numYperChunk x numTtot
Peaks are greater than the peak threshold, and above than the two neighboring digitized points. 
*/

__global__ void CountPeaks_HPD(int16_t* InData, CountDataType* OutData, int32_t* xidMax)
{
	int32_t xid = blockIdx.x * blockDim.x + threadIdx.x;

	int16_t pastPoint, presentPoint, futurePoint;

	if (xid < *xidMax)
	{
		pastPoint = InData[xid];
		presentPoint = InData[xid + 1];
		futurePoint = InData[xid + 2];
	}
	else
	{
		pastPoint = 0;
		presentPoint = 0;
		futurePoint = 0;
	}
	if ((presentPoint >= PEAKTHRESH_HPD_1P) && (presentPoint >= pastPoint) && (presentPoint >= futurePoint))
	{
		if (presentPoint >= PEAKTHRESH_HPD_2P)
		{
			if (presentPoint >= PEAKTHRESH_HPD_3P)
			{
				if (presentPoint >= PEAKTHRESH_HPD_4P)
				{
					if (presentPoint >= PEAKTHRESH_HPD_5P)
					{
						OutData[xid] = 5;
					}
					else
					{
						OutData[xid] = 4;
					}
				}
				else
				{
					OutData[xid] = 3;
				}
			}
			else
			{
				OutData[xid] = 2;
			}
		}
		else
		{
			OutData[xid] = 1;
		}
	}
	else
	{
		OutData[xid] = 0;
	}

}

/*Maximum Shifting: circularly shift each pixel's histogram so that the maximum value is at the first timepoint
See publication - we found that for our setup the most accurate results are obtained when an entire line is summed
together to find the maximum value, and that is set as the t = 0 time. 
*/
__global__ void MaxMinShift_LineShift(AvgDataType* InData, AvgDataType* OutData, int32_t* numX, int32_t* numT, int32_t* numXnumYnumT)
{
	long yid = blockIdx.y * blockDim.y + threadIdx.y;
	long inidx_off = yid * *numT * *numX;
	long tid, xid, MaxIdx = 0;
	AvgDataType MaxVal = InData[inidx_off];
	AvgDataType MaxTemp = 0;

	for (tid = 0; tid < (*numT); tid++)
	{
		MaxTemp = 0;
		for (xid = 0; xid < *numX; xid++)
		{
			MaxTemp += InData[inidx_off + xid * *numT + tid];
		}
		if (MaxTemp > MaxVal)
		{
			MaxIdx = tid;
			MaxVal = MaxTemp;
		}
	}

	for (tid = 0; tid < *numT; tid++)
	{
		for (xid = 0; xid < *numX; xid++)
		{
			OutData[inidx_off + xid * *numT + tid] = (InData[(inidx_off + xid * *numT + tid + MaxIdx) % *numXnumYnumT]);
		}
	}
}


/*Binning
*/
__global__ void FluorescenceDecayBin(AvgDataType* InData, AvgDataType* OutData, int32_t* numX, int32_t* numY, int32_t* numT)
{
	long xid = blockIdx.x * blockDim.x + threadIdx.x;
	long yid = blockIdx.y * blockDim.y + threadIdx.y;
	long tid = blockIdx.z * blockDim.z + threadIdx.z;
	long xidx, yidx;
	AvgDataType outTemp = 0;

	for (xidx = -BINSTEP; xidx <= BINSTEP; xidx++)
	{
		for (yidx = -BINSTEP; yidx <= BINSTEP; yidx++)
		{
			outTemp += InData[((xid + xidx + *numX) % *numX) * *numT + ((yid + yidx + *numY) % *numY) * *numT * *numX + tid]; //exclude edge pixels
		}
	}
	OutData[xid * *numT + yid * *numT * *numX + tid] = outTemp;
}


/*Calculate multiphoton microscopy intensity and lifetime, and phasor components g and s
*/
__global__ void DoingMPMAndFLIMKernel(AvgDataType* InData, float* MPMImage, int32_t* numX, int32_t* numT, float* Thresh, uint16_t* GAv, uint16_t* SAv,
	float* Meanlifetime)
{
	long xid = blockIdx.x * blockDim.x + threadIdx.x;
	long yid = blockIdx.y * blockDim.y + threadIdx.y;
	long tidx;
	long outidx = xid + yid * *numX;
	long inidx = xid * *numT + yid * *numT * *numX;
	float G, S;

	MPMImage[outidx] = 0.0f;
	for (tidx = 0; tidx < *numT; tidx++)
	{
		MPMImage[outidx] += (InData[inidx + tidx]);
	}
	if (MPMImage[outidx] >= *Thresh)
	{
		float SumSine = 0.0f, SumCosine = 0.0f;
		AvgDataType SumSum = 0;
		for (tidx = 0; tidx < DECAYLENGTH; tidx++)
		{
			SumSine += ((float)InData[inidx + tidx] * sinf(TAXIS_COEFF * (tidx)));
			SumCosine += ((float)InData[inidx + tidx] * cosf(TAXIS_COEFF * (tidx)));
			SumSum += (InData[inidx + tidx]);
		}

		G = SumCosine / ((float)SumSum + 1E-15f);
		S = SumSine / ((float)SumSum + 1E-15f);
	}
	else
	{
		G = -1.0f;
		S = 0.0f;
	}
	if (S > 0.0f && G > 0.0f)
	{
		Meanlifetime[outidx] = (LT_COEFF * (S / G));
		SAv[outidx] = (uint16_t)(S * 10000.0f);
		GAv[outidx] = (uint16_t)(G * 10000.0f);
	}
	else
	{
		Meanlifetime[outidx] = 0.0f;
		SAv[outidx] = 0;
		GAv[outidx] = 0;
	}
}
