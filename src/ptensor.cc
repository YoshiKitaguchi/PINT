#include "pint.h"

#include <algorithm>
#include <stdio.h>
#include <cstring>

using namespace pint;

PTensor::PTensor()
{
    _ndim = 1;
    _shape[0] = _shape[1] = _shape[2] = 1;
    _size = sizeof(double);
    _data = (double *)malloc(_size);
}

/*PTensor::PTensor(int ndim, int * shape)
{
    _ndim = min(ndim, 3);

    _size = sizeof(double);
    for (int i = 0; i < _ndim; i++)
    {
        _shape[i] = shape[i];
        _size *= shape[i];
    }

    _data = (double *)malloc(_size);
}*/

PTensor::PTensor(int ndim, const int * shape, double init)
{
    _ndim = min(ndim, 3);

    _size = sizeof(double);
    for (int i = 0; i < _ndim; i++)
    {
        _shape[i] = shape[i];
        _size *= shape[i];
    }

    _data = (double *)malloc(_size);

    for (int i = 0; i < _size; i++)
    {
        _data[i] = init;
    }
}

PTensor::PTensor(const PTensor& t)
{
    this->_ndim = t._ndim;
    this->_size = t._size;

    for (int i = 0; i < _ndim; i++)
    {
        this->_shape[i] = t._shape[i];
    }

    this->_data = (double *)malloc(this->_size);
    std::memcpy(this->_data, t._data, this->_size);
}

PTensor::~PTensor()
{
    free(_data);
}


// Assignment/compound assignment ops

PTensor & PTensor::operator=(const PTensor &rhs)
{
    if (this == &rhs) { return *this; }

    /*if (this->_size != rhs._size)
    {
        printf("Unequal PTensor sizes\n");
        exit(1);
    }*/

    this->_ndim = rhs._ndim;
    this->_size = rhs._size;

    for (int i = 0; i < _ndim; i++)
    {
        this->_shape[i] = rhs._shape[i];
    }

    free(this->_data);
    _data = (double *)malloc(this->_size);
    memcpy(this->_data, rhs._data, this->_size);
    return *this;
}

PTensor & PTensor::operator+=(const PTensor &rhs)
{
    if (this->_size != rhs._size)
    {
        printf("Unequal PTensor sizes\n");
        exit(1);
    }

    for (int i = 0; i < _size; i++) { this->_data[i] += rhs._data[i]; }

    return *this;
}

PTensor & PTensor::operator-=(const PTensor &rhs)
{
    if (this->_size != rhs._size)
    {
        printf("Unequal PTensor sizes\n");
        exit(1);
    }

    for (int i = 0; i < _size; i++) { this->_data[i] -= rhs._data[i]; }

    return *this;
}

PTensor & PTensor::operator*=(const PTensor &rhs)
{
    if (this->_size != rhs._size)
    {
        printf("Unequal PTensor sizes\n");
        exit(1);
    }

    for (int i = 0; i < _size; i++) { this->_data[i] *= rhs._data[i]; }

    return *this;
}

PTensor & PTensor::operator/=(const PTensor &rhs)
{
    if (this->_size != rhs._size)
    {
        printf("Unequal PTensor sizes\n");
        exit(1);
    }

    for (int i = 0; i < _size; i++) { this->_data[i] /= rhs._data[i]; }

    return *this;
}


// Comparison ops

bool PTensor::operator==(const PTensor &rhs) const
{
    if (this == &rhs) { return true; }

    if (this->_size != rhs._size) { return false; }

    for (int i = 0; i < _size; i++)
    {
        if (this->_data[i] != rhs._data[i])
        {
            return false;
        }
    }

    return false;
}

// Dot prod

const PTensor dot(const PTensor &a, const PTensor &b)
{
    
    return a;
}
