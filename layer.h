#ifndef LAYER_H
#define LAYER_H

#include<iostream>
#include "neuron.h"

class Layer
{
public:
    uint32_t num_neu;
    Neuron *neu;

public:
    Layer() = default;
    ~Layer();

    void set_layer(uint32_t number_of_neurons);
};



#endif