#ifndef CNN_H
#define CNN_H

using namespace std;

namespace pint
{
    class ConvolutionalNet
    {
        private:
            PTensor * convLayer;
            vector <PTensor *> filterLayers;
            vector<vector<vector<PTensor *>>> LayersImagePatch;
            int rowLayer;
            int colLayer;
            int numFiltersLayers;
            int filterSize;
        public:
            ConvolutionalNet();
            ~ConvolutionalNet();
            PTensor * getPatch (PTensor * layer, int row, int col, int row_size, int col_size);
            double summation (PTensor *);
            double ReLU(double input);
            void generateFilters (int numFilters, int size );
            vector <PTensor *> convolution(int numLayers, int filterSize, PTensor * layer, string activation);
            vector <PTensor *> convLayer_backPropagation (vector <vector <vector <double>>> dL_dout, double learningRate);
            
            double maximum(PTensor * batch);
            double minimum(PTensor * batch);
            double average(PTensor * batch);
            vector <PTensor *> pool(vector <PTensor *> layers, int * pool_shape, string pool_type);
    };

}

#endif