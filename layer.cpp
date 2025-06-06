#include"layer.h"

using namespace std;

void Layer::set_layer(uint32_t number_of_neurons)
{
	this->num_neu = number_of_neurons;
	this->neu = new Neuron[number_of_neurons]();
    cout << "Created layer with number of neuron " << this->num_neu << endl;
}

Layer::~Layer() 
{
	delete[] this->neu;
    // cout << "Layer destroyed" << endl;
}