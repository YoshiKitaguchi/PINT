#include "pint.h"
#include <iostream>
#include <algorithm>
#include "pint.h"

namespace pint
{
    // constructor of ConvolutionalNet
    ConvolutionalNet::ConvolutionalNet()
    {
        // demo filter layer of 3 x 3
        int shape[3];
        shape[0] = 3;
        shape[1] = 3;
        shape[2] = 1;
        PTensor *defaultLayer = new PTensor(3, shape);
        defaultLayer->at(0,0) = 0; // 1x
        defaultLayer->at(0,1) = 0; // 2x
        defaultLayer->at(0,2) = 0; // 3x
        defaultLayer->at(1,0) = 0; // 1y
        defaultLayer->at(1,1) = 1; // 2y
        defaultLayer->at(1,2) = 0; // 3y
        defaultLayer->at(2,0) = 0; // 1z
        defaultLayer->at(2,1) = 0; // 2z
        defaultLayer->at(2,2) = 0; // 3z
        /*
        default convolution layer
        1 0 1
        0 1 0
        1 0 1
        */
        this->filterLayers.push_back(defaultLayer);
    }

    ConvolutionalNet::~ConvolutionalNet()
    {
        // TODO: reset everything!
    }

    // get the patch (a zoomed up matrix) from a matrix
    PTensor * ConvolutionalNet::getPatch(PTensor * layer, int row, int col,  int row_size, int col_size)
    {
        // set up the patch
        int shape[3];
        shape[0] = row_size;
        shape[1] = col_size;
        shape[2] = 1;
        PTensor * patch = new PTensor(3, shape);

        // get the patch
        for (int i = 0; i < row_size; i++)
        {
            for (int j = 0; j < col_size; j++)
            {
                patch->at(i, j) = layer->at(row + i, col + j);
            }
        }

        return patch;
    }

    // sum every cells in the matrix (batch)
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

    // Relu function
    double ConvolutionalNet:: ReLU(double input)
    {
        // Declare the fully activated map and clone all the values over to it
        return input < 0.0  ? 0.0 : input;
    }

    void ConvolutionalNet::generateFilters (int numFilters, int size ) 
    {
        srand (time(NULL));
        for (int k = 0; k < numFilters; k++)
        {
            int shape[3] = {size, size , 1};
            PTensor * pt = new PTensor (3, shape);
            for (int i = 0; i < size; i++)
            {
                for (int j = 0; j < size; j++)
                {
                    pt->at(i, j) = (double) ((rand()) % 25 - 10.0)  / (size * size);
                }
            }
            filterLayers.push_back(pt);
        }
    }

    // convolution layer
    vector <PTensor *> ConvolutionalNet::convolution(int numFilterLayers, int filterSize, PTensor * layer, string activation)
    {
        // set up the filter layer

        rowLayer = layer->_shape[0];
        colLayer = layer->_shape[1];
        this->numFiltersLayers = numFilterLayers;
        this->filterSize = filterSize;
        if (numFilterLayers == 1) 
        {
            filterSize = 3;
        }
        else if (numFilterLayers > 1)
        {
            filterLayers.clear();
            generateFilters (numFilterLayers, filterSize);
        }
        else {
            exit(1);
        }

        if (filterSize > rowLayer || filterSize > colLayer)
        {
            exit(1);
        }

        vector <PTensor *> convLayers;

        for (int k = 0; k < numFilterLayers; k++) {
            // set up the acitvation map (result after the filter and activation function)
            int shape[3];
            shape[0] = rowLayer - filterSize + 1;
            shape[1] = colLayer - filterSize + 1;
            shape[2] = 1;
            PTensor * activationMap = new PTensor(3, shape);
            this->convLayer = this->filterLayers[k];
            vector <vector <PTensor *>> ImagePatchRow;
            // run filter and activation
            for (int i = 0; i < shape[0]; i++) 
            {
                vector<PTensor *> ImagePatchCol;
                for (int j = 0; j < shape[1]; j++)
                {
                    PTensor * patch = getPatch(layer, i , j, filterSize, filterSize);
                    ImagePatchCol.push_back(patch);
                    PTensor temp = dot(*patch, *(this->convLayer));
                    
                    if (activation == "ReLU")
                    {
                        activationMap->at(i, j) = ReLU(summation(&temp));
                    }
                }
                ImagePatchRow.push_back(ImagePatchCol);
            }
            LayersImagePatch.push_back(ImagePatchRow);
            convLayers.push_back(activationMap);
            // cout << *(convLayers[0]) << endl;
        }
        return convLayers;
    }

    vector <PTensor *> ConvolutionalNet:: convLayer_backPropagation (vector <vector <vector <double>>> dL_dout, double learningRate) 
    {
        vector <PTensor *> dL_dF_params;
        for (int k = 0; k < numFiltersLayers; k++)
        {
            int shape [3] = {filterSize, filterSize, 1};
            PTensor * pt = new PTensor(3, shape);
            for (int i = 0; i < rowLayer ; i++)
            {
                for (int j = 0; j < colLayer; j++)
                {
                    *pt += *(LayersImagePatch[k][i][j]) * dL_dout[k][i][j];
                }
            }
            dL_dF_params.push_back(pt);
            *(filterLayers[k]) -= learningRate* *pt; 
        }
        return dL_dF_params;
    }

    double ConvolutionalNet:: maximum( PTensor * batch )
    {
        // TODO: Update this to the numeric limits equivalent
        double max = INT8_MIN;

        // Iteratre through the batch looking for the maximum
        for(int a = 0; a < batch->_shape[0]; a++)
        {
            for(int b = 0; b < batch->_shape[1]; b++)
            {   
                // Holds the current batch value
                double current = batch->at(a,b);
                // Sets the maximum to the current if the current is greater
                if(current > max) { max = current; }
            }
        }

        // What else are we gonna return?
        return max;
    }

    double ConvolutionalNet::minimum( PTensor * batch )
    {
        // TODO: Update this to the numeric limits equivalent
        double min = INT8_MAX;

        // Iteratre through the batch looking for the maximum
        for(int a = 0; a < batch->_shape[0]; a++)
        {
            for(int b = 0; b < batch->_shape[1]; b++)
            {   
                // Holds the current batch value
                double current = batch->at(a,b);
                // Sets the maximum to the current if the current is greater
                if(current < min) { min = current; }
            }
        }

        // What else are we gonna return?
        return min;
    }

    double ConvolutionalNet::average( PTensor * batch)
    {
        // Sums it all then divides it by the count of tiles in the batch.
        return this->summation(batch)/( (double) ( batch->_shape[0] * batch->_shape[1] ) );
    }

    // This grabs the pool of a the 
    vector <PTensor *> ConvolutionalNet::pool(vector <PTensor *> layers, int * pool_shape, string pool_type)
    {
        int pool_height = layers[0]->_shape[0]/pool_shape[0];
        int pool_width = layers[0]->_shape[1]/pool_shape[1];
        int numlayers = layers.size();

        vector <PTensor *> _poolLayers;
        
        for (int k = 0; k < numlayers; k++)
        {
            PTensor * layer = layers[k];
            // This will be the shape for the new PTensor that will hold the pooled stuff
            int shape[3];
            // The width
            shape[0] = pool_width;
            // The height
            shape[1] = pool_height;
            // TODO: change the count of colour layers
            shape[2] = 1;
            // Make the new tensor with the given shape
            PTensor * _pool = new PTensor(3, shape);

            for (int i = 0; i < shape[0]; i++) 
            {
                for (int j = 0; j < shape[1]; j++)
                {
                    // This is the patch of the given PTensor
                    PTensor * patch = getPatch(layer, i*pool_shape[0], j*pool_shape[1], pool_shape[0], pool_shape[1] );
                    // If the pooling style is a maximum pooling style, call the max function
                    if(pool_type.compare("max") == 0) { _pool->at(i,j) = this->maximum(patch); }
                    // If the pooling style is a minimum pooling style, call the min function
                    else if(pool_type.compare("min") == 0) { _pool->at(i,j) = this->minimum(patch); }
                    // If the pooling style is a average pooling style, call the avg function
                    else if(pool_type.compare("min") == 0) { _pool->at(i,j) = this->average(patch); }
                    // If the function is invalid, exit
                    else { std::cout << "INVALID POOLING FUNCTION!!!!"; exit(1); }
                }
            }
            // Return the pooled layer
            _poolLayers.push_back(_pool);
        }
        return _poolLayers;
    }
}