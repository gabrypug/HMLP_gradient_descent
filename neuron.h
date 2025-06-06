#ifndef NEURON_H
#define NEURON_H

#include<iostream>
#include"Q/quaternions.h"
using namespace std;

class Neuron{
public:
    quaternion actv; 				// Uscita del neurone
	quaternion *out_weights;		// Vettore pesi in uscita dal neurone
	quaternion bias;				// Bias del neurone
	quaternion z;					// Errore associato al neurone

	quaternion dactv;				// Variabile aux
	quaternion *dw;					// Vettore correzione dei pesi
	quaternion *dwp;				// Vettore correzione dei pesi precedente
	quaternion dbias;				// Correzione del bias
	quaternion dbp;					// Correzione del bias predente
	quaternion dz;					// Varaibile aux

public:
    Neuron() = default;
    ~Neuron();

    void set_neuron(uint32_t num_out_weights, bool memory_flag);
    // memory_flag = 1 if we want a neuron with dynamic
    // memory_flag = 0
};

#endif