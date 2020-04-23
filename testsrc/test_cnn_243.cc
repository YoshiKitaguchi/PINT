#include "pint.h"

using namespace pint;

int main()
{
    if (pint::init()) { exit(1); }

    cout << "INPUT:" << endl;
    int shape[3];
    shape[0] = 4;
    shape[1] = 4;
    shape[2] = 1;
    PTensor *input = new PTensor(3, shape);
    input->at(0,0) = input->at(0,1) = input->at(0,2) = input->at(0,3) = 0;
    input->at(1,0) = input->at(1,1) = input->at(1,2) = input->at(1,3) = 1;
    input->at(2,0) = input->at(2,1) = input->at(2,2) = input->at(2,3) = 2;
    input->at(3,0) = input->at(3,1) = input->at(3,2) = input->at(3,3) = 3;
    cout << *input << endl;


    ConvolutionalNet *net = new ConvolutionalNet();

    cout << "convolution Layer with ReLU: " << endl;
    vector <PTensor *> convLayer1 = net->convolution(2, 3, input, "ReLU");
    cout << *(convLayer1[1]) << endl;
    
    cout << "max-pooling: " << endl;
    int shape1[2]  = {2,2};
    vector <PTensor *> maxPool1 = net->pool(convLayer1, shape1, "max");
    cout << *(maxPool1[1]) << endl;
    return 0;
}
