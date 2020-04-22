#ifndef CNN_H
#define CNN_H

using namespace std;

namespace pint
{
    class ConvolutionalNet
    {
        private:
            PTensor * convLayer;
            string activationFunc;
            string pooling;
        public:
            ConvolutionalNet();
            ~ConvolutionalNet();
            PTensor * getPatch (PTensor * layer, int row, int col, int size);
            double summation (PTensor *);
            PTensor * convolution(PTensor * layer, string activation);
    };

}

#endif