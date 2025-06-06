#include"neuron.h"


void Neuron::set_neuron(uint32_t num_out_weights, bool memory_flag)
{
    this->actv = q_set();
	this->out_weights = new quaternion[num_out_weights](); 
	this->bias= q_set();
	this->z = q_set();

	this->dactv = q_set();
	this->dw = new quaternion[num_out_weights]();
    if(memory_flag){

        // cout << "Neuron setted with memory" << endl;
        this->dwp = new quaternion[num_out_weights]();
        for(size_t i = 0; i < num_out_weights; i++){
        this->dwp[i] = q_set();
        }
        this->dbp = q_set();

    } else {
        // cout << "Neuron setted without memory" << endl;
        this->dwp = nullptr;
    }
    

	this->dbias = q_set();
	this->dz = q_set();
}

Neuron::~Neuron() 
{    
	delete[] this->out_weights;
    delete[] this->dw;
    if(this->dwp != nullptr) delete[] this->dwp;
    // cout << "Neuron destroyed" << endl;
}