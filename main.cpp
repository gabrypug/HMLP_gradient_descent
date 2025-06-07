#include<iostream>
#include "hmlpl.h"

using namespace std;


int main(int argc, char *argv[]) {

    uint32_t n_h = 5;
    uint32_t n_th = 3;
    HMLPL net;
    uint32_t num_layers{3};
    uint32_t num_hidden_neurons[num_layers-2] = {n_h};
    double rate = 0.05;
    double momentum = 0.01;
    uint32_t num_thread = n_th;
    uint32_t num_epochs = 10000;
    net.set_HMLPL(0,num_layers,num_hidden_neurons,rate,momentum,0,num_thread,num_epochs, 0.000001, 0.00000001);
    net.get_training_data("dati/HL_input.txt","dati/HL_target.txt");
    net.init();

    // TRAINING
    net.train_neural_net();
    net.export_weight_vector();

    if (net.upload_weights_bias("dati/weights.txt","dati/bias.txt")) {
        net.get_inputs_test("dati/HL_pred.txt");
        net.test();
    }

    return 0;
}
