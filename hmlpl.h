#ifndef HMLPL_H
#define HMLPL_H

#include<iostream>
#include<fstream>
#include<pthread.h>
#include<iomanip>
#include<unistd.h>
#include<float.h>
#include<cmath>
#include"constant.h"
#include"impexp.h"
#include"hmlp.h"


//==========================================================
typedef struct t_arg{
    u_int aux_Id;
} t_arg;
//==========================================================

class HMLPL
{
public:
    uint32_t id;
    bool mem_opt;
	uint32_t num_layers;
    Layer *lay;
    uint32_t *num_neurons;
    uint32_t num_weight;
    uint32_t num_bias;
    uint32_t mode;
    uint32_t num_samples;
    double **inputs;
    double **targets;    
    quaternion *cost;
    uint32_t train_iter;
    uint32_t actual_epoch;
    quaternion component_error;
    double actual_error;
    double goal_error;
    double gradient;
    double min_gradient;
    uint32_t num_epochs;
    bool end_flag;
    double alpha;
    double momentum;
    double **weight_vector;
    double **error_vector;
    double *tmp_weight_vector;
    double *gradient_vector; // It is initialized into the weight_vector_init()

    // AUX MANAGING
    uint32_t num_aux;
    HMLP *aux;
    t_arg* arg;

    // PARALLELIZATION CONTROL
    uint32_t launch_epoch;
    uint32_t update_epoch;
    int32_t token;

    // PERFORMANCE COMPARISION
    double launch_error;

    // OUTPUT GENERATION
    ofstream File, reportFile;

    // Creator and Destructor
    HMLPL();
    ~HMLPL();

    // Member functions for creation
    void set_HMLPL(uint32_t id, uint32_t num_layers, uint32_t* num_hidden_neurons,
        double alpha, double momentum, uint32_t mode, uint32_t num_aux, uint32_t num_epochs, 
        double goal_error, double min_gradient);
    int init();
    int create_architecture(void);
    int initialize_weights(void);

    // Members functions for reading from file
    int get_training_data(const std::string inputs_file, const std::string targets_file);
    int get_inputs(const std::string inputs_file);
    int get_targets(const std::string targets_file);
    int get_inputs_test(const std::string inputs_file);
    void normalize_training_vectors();
    void denormalize_training_vectors();

    // Members functions for training
    void train_neural_net(void);
    void train_seq_mode(void);
    void train_batch_mode(void);

    void feed_input(uint32_t i);
    void forward_prop(void);
    void back_prop(uint32_t p);
    void back_prop_batch(uint32_t p);
    void update_weights(void);
    void update_weights_batch(void);
    void update_predicted_weights(void);
    void init_corrections();
    
    // Weight vector managing
    void weight_vector_init(void);
    void weight_vector_update(uint32_t actual_epoch);
    void export_weight_vector(void);
    void weight_vector_restore(void);
    bool download_weights_bias(void);
    bool upload_weights_bias(const std::string weights_file, const std::string bias_file);

    // Members for evaluation of performances
    void compute_cost(uint32_t i);
    void compute_grad(uint32_t p);
    bool gradient_evaluation(uint32_t weight_Id);
    bool performance_evaluation(void);

    // Members for testing
    void test(void);
    void feed_input_test(uint32_t i);
    void forward_prop_test(void);

    // Members for prediction
    void predict(uint32_t num_step, uint32_t input_step); // da utilizzare quando il numero di neuroni output è uguale al numero di neuroni input
    void feed_input_predict(uint32_t i);
    void forward_prop_predict(void);

    // Members for Aux creation
    int create_aux(void);

    // Members for aux launch
    void start_aux(void); 
    void aux_return(void);  
    
};

// Thread function
void *aux_thread(void *arg);

#endif 