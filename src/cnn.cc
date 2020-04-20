#include "pint.h"
#include <iostream>
#include <algorithm>
#include "ptensor.h"

namespace pint
{
    ConvolutionalNet::ConvolutionalNet()
    {
        int shape[3];
        shape[0] = 3;
        shape[1] = 3;
        shape[2] = 1;
        PTensor *defaultLayer = new PTensor(3, shape);
        defaultLayer->at(0,0) = 1; // 1x
        defaultLayer->at(0,1) = 0; // 2x
        defaultLayer->at(0,2) = 1; // 3x
        defaultLayer->at(1,0) = 0; // 1y
        defaultLayer->at(1,1) = 1; // 2y
        defaultLayer->at(1,2) = 0; // 3y
        defaultLayer->at(2,0) = 1; // 1z
        defaultLayer->at(2,1) = 0; // 2z
        defaultLayer->at(2,2) = 1; // 3z
        /*
        default convolution layer
        1 0 1
        0 1 0
        1 0 1
        */
        this->convLayer = defaultLayer;
        // more varible initializations

        this->activationFunc = "ReLU";
        this->pooling = "maxPool";
    }

    ConvolutionalNet::~ConvolutionalNet()
    {
        // TODO: reset everything!
    }

    PTensor * ConvolutionalNet::getPatch(PTensor * layer, int row, int col, int size)
    {
        int shape[3];
        shape[0] = size;
        shape[1] = size;
        shape[2] = 1;
        PTensor * patch = new Ptensor(3, shape);

        for (int i = 0; i < row; i++)
        {
            for (int j = 0; j < col; j++)
            {
                patch->at(i, j) = layer->at(row + i, col + j);
            }
        }

        return patch;
    }

    double ConvolutionalNet:: summation (PTensor * batch) {
        double sum = 0;
        int size = batch->_shape[0];
        for (int i = 0; i < size; i++) {
            for (int j = 0; j < size; j++) {
                sum += batch->at(i,j);
            }
        }
        return sum;
    }

    PTensor * ConvolutionalNet::convolution(PTensor * layer, string activation)
    {
        int filterSize = this->convLayer->_shape[0];
        int rowLayer = layer->_shape[0];
        int colLayer = layer->_shape[1];
        if (filterSize > rowLayer || filterSize > colLayer)
        {
            exit(1);
        }
        int shape[3];
        shape[0] = rowLayer - filterSize + 1;
        shape[1] = colLayer - filterSize + 1;
        shape[2] = 1;
        PTensor * activationMap = new Ptensor(3, shape);
        for (int i = 0; i < shape[0]; i++) 
        {
            for (int j = 0; j < shape[1]; j++)
            {
                PTensor * patch = getPatch(layer, i, j, row);
                activationMap->at(i, j) = summation(dot(patch, this->convLayer));
            }
        }

        return activationMap;
    }
}