#ifndef HMLP_H
#define HMLP_H

#include<iostream>
#include<pthread.h>
#include"layer.h"

class HMLP
{
    public:
    uint32_t id;
    pthread_t tid;
    bool mem_opt;
	uint16_t num_layers;
    Layer *lay;
    uint32_t *num_neurons;
    uint32_t num_weight;
    uint32_t num_samples;
    double **inputs;
    double **targets;    
    quaternion *cost;
    uint32_t train_iter;
    uint32_t actual_epoch;
    double actual_error;
    double gradient;
    double min_gradient;
    uint32_t num_epochs;
    bool end_flag;
    double alpha;
    double momentum;
    double **weight_vector;
    double **error_vector;
    double norm_min;
    double norm_max;
    uint32_t start_epoch;
    uint32_t step;
    uint32_t num_work;
    uint32_t* work_vector;
    double max_t;
    double min_t;


public:
    HMLP();
    ~HMLP();


    // Member functions for creation
    void set_HMLP(uint32_t id, uint32_t num_layers, uint32_t* num_hidden_neurons,
        double alpha, double momentum, uint32_t num_epochs, double min_gradient);
    int init();
    int create_architecture();
    int initialize_weights();

    // Members functions for reading from file
    void nomalize_targets(uint32_t weight_Id);

    // Members functions for training
    void train_neural_net(void);
    void train_neural_net(uint32_t weight_Id);
    void feed_input(uint32_t i);
    void forward_prop(void);
    void back_prop(uint32_t p,uint32_t weight_Id);
    void update_weights(void);

    // Members for evaluation of performances
    void compute_cost(uint32_t i,uint32_t weight_Id);
    void compute_grad(void);

    // Members for testing
    quaternion predict_weight(uint32_t predicted_epoch);

    // Member for the resetting
    void reset_net(uint32_t net_indx);

};

double find_max(double** vector, int size1_start, int size1_end, int size2);
double find_min(double** vector, int size1_start, int size1_end, int size2);

#endif 