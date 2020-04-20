#ifndef CNN_H
#define CNN_H

#include "pint.h"

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


    }


}

#endif