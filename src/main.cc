#include "pint.h"

#include <iostream>

//using namespace pint;

int main()
{
    if (pint::init()) { exit(1); }

    printf("Hello, world!\n");

    pint::OpGraph g;
    //Add two nodes
    /*g.addNode("a")->addNode("b")->addNode("+","a","b",dAdd)->setNode("a",3)->setNode("b",4);

    //Multiply result of node with constant
    g.addNode("c")->addNode("*","+","c",dMult)->setNode("c",5)->setNode("+",2);

    //Evaluate it
    cout << g.evalNode("*") <<endl;
    */return 0;
}
