#include<unistd.h>
#include<float.h>
#include<cmath>
#include"constant.h"
#include"impexp.h"
#include"hmlp.h"

using namespace std;

//bool flag = false;

HMLP::HMLP(){
    this->id = 0;
    this->mem_opt = true;
    this->num_layers = 0;
    this->lay = nullptr;
    this->num_neurons = nullptr;
    this->num_weight = 0;
    this->num_samples = 0;
    this->inputs = nullptr;
    this->targets = nullptr;
    this->cost = nullptr;
    this->train_iter = 0;
    this->actual_epoch = 0;
    this->actual_error = 0;
    this->gradient = 0;
    this->min_gradient = 0;
    this->num_epochs = 0;
    this->end_flag = false;
    this->alpha = 0;
    this->momentum = 0;
    this->weight_vector = nullptr;
    this->error_vector = nullptr;
    this->norm_min = 0.3;
    this->norm_max = 0.4;
    this->start_epoch = 0;
    this->step = 0;
    this->num_work = 0;
    this->work_vector = nullptr;
    this->max_t = 0;
    this->min_t = 0;
}

HMLP::~HMLP(){
    if(this->lay != nullptr) delete[] this->num_neurons;
    if(this->lay != nullptr) delete[] this->lay;
    if(this->cost != nullptr) delete[] this->cost;

    if(this->targets != nullptr){
        for(size_t i = 0; i < 4; i++){
            delete[] this->targets[i];
        }
        delete[] this->targets;
    }

    if(this->work_vector != nullptr ) delete[] this->work_vector;
    
    cout << ">> AUX#" << this->id+1 << " destroyed <<" << endl;
}

//==================================================================================
// FUNCTIONS FOR CREATION
//==================================================================================

void HMLP::set_HMLP(uint32_t id, uint32_t num_layers, uint32_t* num_hidden_neurons,
        double alpha, double momentum, uint32_t num_epochs, double min_gradient){
    this->id = id;
    this->num_layers = num_layers;

    this->num_neurons = new uint32_t[this->num_layers]();
    for(size_t i = 1; i < this->num_layers-1; i++){
        this->num_neurons[i] = num_hidden_neurons[i-1];
        // cout << "AUX#" << this->id << " | layer#" << i+1 << " has " << this->num_neurons[i] << " neurons" << endl;
    }
    this->alpha = alpha;
    this->momentum = momentum;
    this->num_epochs = num_epochs;
    this->min_gradient = min_gradient;
}

int HMLP::init(){
    if(create_architecture() != SUCCESS_CREATE_ARCHITECTURE)
    {
        cerr << "ERROR: in creating architecture..." << endl;
        return ERR_INIT;
    }

    cout <<  endl;
    cout << "MLP CREATED SUCCESSFULLY..." << endl;
    cout <<  endl;

    return SUCCESS_INIT;
}

int HMLP::create_architecture()
{

    this->lay = new Layer[this->num_layers]();

    for(size_t i=0; i < this->num_layers; i++)
    {
        this->lay[i].set_layer(this->num_neurons[i]);

        cout << "Created Layer: "<< i+1 << endl;
        cout << "Number of Neurons in Layer " << i+1 << ": " << this->lay[i].num_neu << endl;

        for(size_t j=0; j < this->num_neurons[i]; j++)
        {
            if(i < this->num_layers-1) 
            {
                this->lay[i].neu[j].set_neuron(this->num_neurons[i+1], this->mem_opt);
            } 
        }
        printf("\n");
    }

    printf("\n");

    // Initialize the weights
    if(initialize_weights() != SUCCESS_INIT_WEIGHTS)
    {
        cerr << "ERROR: wrong weights initilization..." << endl;
        return ERR_CREATE_ARCHITECTURE;
    }

    this->cost = new quaternion[this->num_neurons[this->num_layers-1]]();
    for(size_t i = 0; i < this->num_neurons[this->num_layers-1]; i++){
        this->cost[i] = q_set();
    }

    return SUCCESS_CREATE_ARCHITECTURE;
}

int HMLP::initialize_weights()
{
    srand(time(0));
    if(this->lay == NULL)
    {
        cerr << "ERROR: No layers in Neural Network" << endl;
        return ERR_INIT_WEIGHTS;
    }

    cout << "Initializing weights" << endl;

    this->num_weight = 0;

    for(size_t i = 0; i < this->num_layers-1; i++)
    {
        
        for(size_t j = 0; j < this->num_neurons[i]; j++)
        {
            for(size_t k = 0; k < this->num_neurons[i+1]; k++)
            {
                // Initialize Output Weights for each neuron
                
                this->lay[i].neu[j].out_weights[k] = q_set( (((double)rand())/((double)RAND_MAX))*2.0 - 1.0,
                                                            (((double)rand())/((double)RAND_MAX))*2.0 - 1.0,
                                                            (((double)rand())/((double)RAND_MAX))*2.0 - 1.0,
                                                            (((double)rand())/((double)RAND_MAX))*2.0 - 1.0);
                this->num_weight++;                                            
                cout << this->num_weight << ") w[" << i << "][" << j << "]["<< k << "]:" << this->lay[i].neu[j].out_weights[k] << endl;
                this->lay[i].neu[j].dw[k] = q_set();

            }

            if(i>0) 
            {
                this->lay[i].neu[j].bias = q_set( (((double)rand())/((double)RAND_MAX))*2.0 - 1.0,
                                                    (((double)rand())/((double)RAND_MAX))*2.0 - 1.0,
                                                    (((double)rand())/((double)RAND_MAX))*2.0 - 1.0,
                                                    (((double)rand())/((double)RAND_MAX))*2.0 - 1.0);
            }
        }

    
    }   
    
    
    for (size_t j=0; j < this->num_neurons[this->num_layers-1]; j++)
    {
        this->lay[this->num_layers-1].neu[j].bias = q_set( (((double)rand())/((double)RAND_MAX))*2.0 - 1.0,
                                                            (((double)rand())/((double)RAND_MAX))*2.0 - 1.0,
                                                            (((double)rand())/((double)RAND_MAX))*2.0 - 1.0,
                                                            (((double)rand())/((double)RAND_MAX))*2.0 - 1.0);
    }

    

    return SUCCESS_INIT_WEIGHTS;
}


void HMLP::reset_net(uint32_t num_samples){
    
    // Reinizializzazione dei parametri di rete
    this->train_iter = 0;
    this->gradient = 0;

    for(size_t i = 0; i < this->num_layers - 1; i++)
    {
        for(size_t j = 0; j < this->num_neurons[i]; j++)
        {
            for(size_t k = 0; k < this->num_neurons[i+1]; k++)
            {
                // Initialize Output Weights for each neuron
                this->lay[i].neu[j].out_weights[k] = q_set( (((double)rand())/((double)RAND_MAX))*2.0 - 1.0,
                                                            (((double)rand())/((double)RAND_MAX))*2.0 - 1.0,
                                                            (((double)rand())/((double)RAND_MAX))*2.0 - 1.0,
                                                            (((double)rand())/((double)RAND_MAX))*2.0 - 1.0);
                this->lay[i].neu[j].dw[k] = q_set();
            }

            if(i>0) 
            {
                this->lay[i].neu[j].bias = q_set( (((double)rand())/((double)RAND_MAX))*2.0 - 1.0,
                                                    (((double)rand())/((double)RAND_MAX))*2.0 - 1.0,
                                                    (((double)rand())/((double)RAND_MAX))*2.0 - 1.0,
                                                    (((double)rand())/((double)RAND_MAX))*2.0 - 1.0);
            }
        }
    }   

    for (size_t j = 0; j < this->num_neurons[this->num_layers - 1]; j++)
    {
        this->lay[this->num_layers - 1].neu[j].bias = q_set( (((double)rand())/((double)RAND_MAX))*2.0 - 1.0,
                                                            (((double)rand())/((double)RAND_MAX))*2.0 - 1.0,
                                                            (((double)rand())/((double)RAND_MAX))*2.0 - 1.0,
                                                            (((double)rand())/((double)RAND_MAX))*2.0 - 1.0);
    }
    this->num_samples = num_samples;

    if(this->targets == nullptr){
        this->targets = new double*[4]();
        for(size_t i = 0; i < 4; i++)
            this->targets[i] = new double[this->num_samples](); 
    }
    else
        for(size_t i = 0; i < 4; i++){
            delete[] this->targets[i];
            this->targets[i] = new double[this->num_samples]();
        }

    

    //cout << "<<<<<<<<<<<<<<<<<<<<<< NUM_SAM =" << this->num_samples << ">>>>>>>>>>>>>>>>>>>>>>>>" << endl;
}

//==============================================================================
// INPUT & OUTPUT
//==================================================================================


void HMLP::nomalize_targets(uint32_t weight_Id){
    double tmp;
    int r,s;
    r = 0;

    this->max_t = find_max(this->weight_vector, (weight_Id * 4), (weight_Id + 1) * 4, this->num_samples);
    this->min_t = find_min(this->weight_vector, (weight_Id * 4), (weight_Id + 1) * 4, this->num_samples);

    for(size_t i = (weight_Id * 4);  i < (weight_Id + 1) * 4; i++){
        s = 0;
        for(size_t j = this->start_epoch; j < this->num_samples; j += this->step) {
            //cout << i << ") " << this->weight_vector[i][j] << endl;
            tmp = this->weight_vector[i][j] - min_t;
            this->targets[r][s] = tmp/(max_t - min_t) * norm_max + norm_min;
            // cout << s << ") " << targets[r][s] << endl;
            s++;
        }
        r++;
    }
    this->train_iter = s;
}

double find_max(double** vector, int size1_start, int size1_end, int size2){
    double max_value{-(double)FLT_MAX};
    for(size_t i = size1_start; i < size1_end; i++)
        for(size_t j = 0; j < size2; j++)
            if(vector[i][j] > max_value) max_value = vector[i][j];
    return max_value;
}
double find_min(double** vector, int size1_start, int size1_end,  int size2){
    double min_value{(double)FLT_MAX};
    for(size_t i = size1_start; i < size1_end; i++)
        for(size_t j = 0; j < size2; j++)
            if(vector[i][j] < min_value) min_value = vector[i][j];
    return min_value;
}

//==================================================================================
// FUNCTIONS FOR TRAINING
//==================================================================================

void HMLP::train_neural_net(uint32_t weight_Id){
    cout << " TRAIN " << endl;

    nomalize_targets(weight_Id);
    
    for(size_t it = 0; it < this->num_epochs; it++)
    {
        actual_error = 0.0;
        for(size_t i = 0; i < this->train_iter; i++)
        {
            feed_input(i);
            forward_prop();
            compute_cost(i,weight_Id);
            back_prop(i,weight_Id);
            update_weights();
        }
        
    }
    cout << "Weight[" << weight_Id << "]Actual_Error = " << cost[0] << endl;
}

void HMLP::feed_input(uint32_t i)
{
    //cout << endl << "<<<<<<F E E D I N P U T>>>>>>" << endl << endl;
    this->lay[0].neu[0].actv = q_set(( (i*this->step + this->start_epoch) * (norm_max / (this->num_samples)) + norm_min),0,0,0);
    // cout << "INPUT[" << i << "] = " << this->lay[0].neu[0].actv << endl;
}

void HMLP::forward_prop(void)
{
    // cout << endl << "<<<<<<F O R W A R D P R O P>>>>>>" << endl << endl;
    quaternion delta_x_w;

    for(size_t i = 1; i < num_layers; i++)
    {
        for(size_t j = 0; j < num_neurons[i]; j++)
        {
            lay[i].neu[j].z = lay[i].neu[j].bias;

            for(size_t k = 0; k < num_neurons[i-1]; k++)
            {
                delta_x_w = q_prod(lay[i-1].neu[k].out_weights[j],lay[i-1].neu[k].actv);
                lay[i].neu[j].z  = q_sum(lay[i].neu[j].z, delta_x_w);
            }
            lay[i].neu[j].actv = q_sigmoid(lay[i].neu[j].z);
        }
    }

}

// Compute Total Cost
void HMLP::compute_cost(uint32_t i,uint32_t weight_Id)
{
    //cout << endl << "<<<<<<C O M P U T E C O S T>>>>>>" << endl << endl;

    quaternion tmpcost;
    quaternion target;
    double tcost = 0;

    for(size_t j = 0; j < this->num_neurons[this->num_layers-1]; j++)
    {
        target = q_set(this->targets[0][i],this->targets[1][i],this->targets[2][i],this->targets[3][i]);        
        tmpcost = q_diff(target, this->lay[this->num_layers-1].neu[j].actv);


        this->cost[j] = q_set(tmpcost.r*tmpcost.r,
                                tmpcost.i*tmpcost.i,
                                tmpcost.j*tmpcost.j,
                                tmpcost.k*tmpcost.k);
                    
        tcost = this->cost[j].r/2 + this->cost[j].i/2 + this->cost[j].j/2 + this->cost[j].k/2;
    }
    // cout << "Weight[" << weight_Id << "][" << i << "] = " << tcost << endl;
    actual_error = ((actual_error * i) + tcost)/(i+1); //_________T O  F I X________   
}


// Back Propogate Error
void HMLP::back_prop(uint32_t p,uint32_t weight_Id)
{
    int j4;
    quaternion error, tar;
    // cout << endl << "<<<<<<B A C K P R O P>>>>>>" << endl << endl;

    for(size_t i = 1; i < this->num_layers; i++){
        for(size_t j = 0; j < this->num_neurons[i]; j++){
            this->lay[i].neu[j].dactv = q_set();
        }
    }

    // <<<<<<Output Layer - from here>>>>>>>>>>
    for(size_t j = 0; j < this->num_neurons[this->num_layers-1]; j++) // j = number of neurons in the output layer
    {
        // ERROR IN THE OUTPUT LAYER
        tar = q_set(this->targets[0][p],this->targets[1][p],this->targets[2][p],this->targets[3][p]);
        error = q_diff(tar,this->lay[this->num_layers-1].neu[j].actv);

        this->lay[this->num_layers-1].neu[j].dz = q_set(error.r * this->lay[this->num_layers-1].neu[j].actv.r * (1 - this->lay[this->num_layers-1].neu[j].actv.r),
                                                error.i * this->lay[this->num_layers-1].neu[j].actv.i * (1 - this->lay[this->num_layers-1].neu[j].actv.i),
                                                error.j * this->lay[this->num_layers-1].neu[j].actv.j * (1 - this->lay[this->num_layers-1].neu[j].actv.j),
                                                error.k * this->lay[this->num_layers-1].neu[j].actv.k * (1 - this->lay[this->num_layers-1].neu[j].actv.k));

        for(size_t k = 0; k < this->num_neurons[this->num_layers-2]; k++) // k = number of neurons in the output-1 layer
        {
            this->lay[this->num_layers-2].neu[k].dw[j] = q_prod(this->lay[this->num_layers-1].neu[j].dz,this->lay[this->num_layers-2].neu[k].actv);
            this->lay[this->num_layers-2].neu[k].dactv = q_sum(this->lay[this->num_layers-2].neu[k].dactv,q_prod(q_conj(this->lay[this->num_layers-2].neu[k].out_weights[j]),this->lay[this->num_layers-1].neu[j].dz));
        }

        this->lay[this->num_layers-1].neu[j].dbias = this->lay[this->num_layers-1].neu[j].dz;
    }
    // Output Layer - to here

    // <<<<<<<<<<<Hidden Layers - from here>>>>>>>>>>>>>
    for(size_t i = this->num_layers-2; i > 0; i--) // i = number of the considerend layer
    {
        for(size_t j = 0; j < this->num_neurons[i]; j++) // j = number of neurons in the considered layer
        {
            tar = q_diffsig(this->lay[i].neu[j].actv);

            this->lay[i].neu[j].dz.r = this->lay[i].neu[j].dactv.r * tar.r;
            this->lay[i].neu[j].dz.i = this->lay[i].neu[j].dactv.i * tar.i;
            this->lay[i].neu[j].dz.j = this->lay[i].neu[j].dactv.j * tar.j;
            this->lay[i].neu[j].dz.k = this->lay[i].neu[j].dactv.k * tar.k;

            for(size_t k = 0; k < this->num_neurons[i-1]; k++) // k = number of neurons in the layer i-1
            {
                this->lay[i-1].neu[k].dw[j] = q_prod(this->lay[i].neu[j].dz,q_conj(this->lay[i-1].neu[k].actv));

                if(i > 1) // if i = 1 we are considering the second layer
                {
                    this->lay[i-1].neu[k].dactv = q_sum(this->lay[i-1].neu[k].dactv,q_prod(q_conj(this->lay[i-1].neu[k].out_weights[j]),this->lay[i].neu[j].dz));
                }
            }

            this->lay[i].neu[j].dbias = this->lay[i].neu[j].dz;
        }
    }
}

void HMLP::update_weights(void)
{
    // cout << endl << "<<<<<<U P D A T E>>>>>>" << endl << endl;
    quaternion correction;

    for(size_t i = 0; i < this->num_layers-1; i++)
    {
        for(size_t j = 0; j < this->num_neurons[i]; j++)
        {
            for(size_t k = 0; k < this->num_neurons[i+1]; k++)
            {
                // Update Weights
                    correction = q_sum(q_prod(this->alpha, this->lay[i].neu[j].dw[k]),q_prod(this->momentum, this->lay[i].neu[j].dwp[k]));
                    this->lay[i].neu[j].out_weights[k] = q_sum(this->lay[i].neu[j].out_weights[k],correction);
                    this->lay[i].neu[j].dwp[k] = correction;
            }

            // Update Bias
            correction = q_sum(q_prod(this->alpha, this->lay[i].neu[j].dbias),q_prod(this->momentum, this->lay[i].neu[j].dbp));
            this->lay[i].neu[j].bias = q_sum(this->lay[i].neu[j].bias, correction);
            this->lay[i].neu[j].dbp = correction;
        }
    }
    // Update bias output neurons
    for(size_t j = 0; j < this->num_neurons[num_layers-1]; j++)
        {
            correction = q_sum(q_prod(this->alpha, this->lay[num_layers-1].neu[j].dbias),q_prod(this->momentum, this->lay[num_layers-1].neu[j].dbp));
            this->lay[num_layers-1].neu[j].bias = q_sum(this->lay[num_layers-1].neu[j].bias, correction);
            this->lay[num_layers-1].neu[j].dbp = correction;

        }

}

//==================================================================================
// FUNCTIONS FOR PREDICTION
//==================================================================================

quaternion HMLP::predict_weight(uint32_t predicted_epoch)
{
    quaternion tmp;
    //cout << "P R E D I C T I O N" << endl;
    this->lay[0].neu[0].actv = q_set(0.80, 0, 0, 0);

    /*
    this->lay[0].neu[0].actv = q_set(predicted_epoch * (norm_max / num_samples) + norm_min, 0, 0, 0);
    cout << predicted_epoch << endl;
    cout << num_samples << endl;
    cout << this->lay[0].neu[0].actv << endl;
    */
    quaternion delta_x_w;

    for(size_t i = 1; i < this->num_layers; i++)
    {   
        for(size_t j = 0; j < this->num_neurons[i]; j++)
        {
            this->lay[i].neu[j].z = this->lay[i].neu[j].bias;

            for(size_t k = 0; k < this->num_neurons[i-1]; k++)
            {
                delta_x_w = q_prod(this->lay[i-1].neu[k].out_weights[j],this->lay[i-1].neu[k].actv);
                this->lay[i].neu[j].z  = q_sum(this->lay[i].neu[j].z, delta_x_w);
            }
            
            this->lay[i].neu[j].actv = q_sigmoid(this->lay[i].neu[j].z);
            /*
            if (i == this->num_layers - 1)
                cout << "OUTPUT[" << j+1 << "] = " << this->lay[i].neu[j].actv << endl;
            */
        }
    }

    tmp = q_set((((this->lay[this->num_layers-1].neu[0].actv.r - norm_min) / norm_max ) * (this->max_t - this->min_t)) + this->min_t,
                (((this->lay[this->num_layers-1].neu[0].actv.i - norm_min) / norm_max ) * (this->max_t - this->min_t)) + this->min_t,
                (((this->lay[this->num_layers-1].neu[0].actv.j - norm_min) / norm_max ) * (this->max_t - this->min_t)) + this->min_t,
                (((this->lay[this->num_layers-1].neu[0].actv.k - norm_min) / norm_max ) * (this->max_t - this->min_t)) + this->min_t);

    return tmp;
}