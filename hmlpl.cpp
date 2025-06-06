
#include"hmlpl.h"

using namespace std;

HMLPL* main1;

pthread_mutex_t Wv, Wt, Wg;

HMLPL::HMLPL(){

    this->id = 0;
    this->mem_opt = true;
    this->num_layers = 0;
    this->lay = nullptr;
    this->num_neurons = nullptr;
    this->num_weight = 0;
    this->num_bias = 0;
    this->num_samples = 0;
    this->inputs = nullptr;
    this->targets = nullptr;
    this->cost = nullptr;
    this->train_iter = 1;
    this->actual_epoch = 0;
    this->component_error = q_set();
    this->actual_error = 0;
    this->gradient = 0;
    this->min_gradient = 0;
    this->num_epochs = 0;
    this->end_flag = false;
    this->alpha = 0;
    this->momentum = 0;
    this->weight_vector = nullptr;
    this->error_vector = nullptr;
    this->tmp_weight_vector = nullptr;
    this->gradient_vector = nullptr;
    this->num_aux = 0;
    this->aux = nullptr;
    this->arg = nullptr;
    this->launch_epoch = 0;
    this->update_epoch = 0;
    this->token = -1;

    // MUTEX INIT
    pthread_mutex_init(&Wv, NULL);
    pthread_mutex_init(&Wt, NULL);
    pthread_mutex_init(&Wg, NULL);
}

HMLPL::~HMLPL(){

    cout << "================== D E S T R U C T O R ====================" << endl;

    if(this->lay != nullptr) delete[] this->num_neurons;
    if(this->lay != nullptr) delete[] this->lay;
    if(this->cost != nullptr) delete[] this->cost;
    cout << "1] HMLP#" << this->id << " DESTRUCTOR: architecture destroyed . . ." << endl;

    if(inputs != nullptr) { 
        freeMatrix(inputs,num_samples);
        inputs = nullptr;
    }
    if(targets != nullptr) {
        freeMatrix(targets, num_samples);
        targets = nullptr;
    }
    cout << "2] HMLP#" << this->id << " DESTRUCTOR: inputs destroyed . . ." << endl;

    if(this->weight_vector != nullptr){
        for(size_t i = 0; i < this->num_weight; i++){
            delete[] this->weight_vector[i];
        }
        delete[] this->weight_vector;
    }
    delete[] this->tmp_weight_vector;
    delete[] this->gradient_vector;
    cout << "3] HMLP#" << this->id << " DESTRUCTOR: weight arrays destroyed . . ." << endl;

    // Remember always to:
    // 1) join the alive threads
    // 2) destroy the mutex
    // 3) delete aux[]
    // in this order
    if(token > 0){
        for(size_t i = 0; i < this->num_aux ; i++)
            if (!pthread_cancel(this->aux[i].tid)){
                cout << ">> Thread " << i+1 << " of " << this->num_aux << " canceled <<" << endl;
            }
        cout << "4] HMLP#" << this->id << " DESTRUCTOR: all the aux nets are joined . . ." << endl;
    }

    pthread_mutex_destroy(&Wv);
    pthread_mutex_destroy(&Wt);
    pthread_mutex_destroy(&Wg);
    cout << "5] HMLP#" << this->id << " DESTRUCTOR: mutex destroyed . . ." << endl;

    File.close();
    cout << "6] MLP#" << " DESTRUCTOR: output file closed . . ." << endl;
 
    if(this->aux != nullptr && this->arg != nullptr){
        delete[] arg;
        delete[] aux;
        cout << "7] HMLP#" << this->id << " DESTRUCTOR: auxs destroyed and deleted . . ." << endl;
    }
    cout << "8] HMLP#" << this->id << " DESTRUCTOR: whole net destroyed." << endl;
}

//==================================================================================
// FUNCTIONS FOR CREATION
//==================================================================================

void HMLPL::set_HMLPL(uint32_t id, uint32_t num_layers, uint32_t* num_hidden_neurons,
        double alpha, double momentum, uint32_t mode, uint32_t num_aux, uint32_t num_epochs, double goal_error, double min_gradient){

    main1 = this;
    this->id = id;
    this->num_layers = num_layers;

    this->num_neurons = new u_int[this->num_layers]();
    for(size_t i = 1; i < this->num_layers-1; i++){
        this->num_neurons[i] = num_hidden_neurons[i-1];
        cout << "MLP#" << this->id << " | layer#" << i+1 << " has " << this->num_neurons[i] << " neurons" << endl;
    }

    this->alpha = alpha;
    this->momentum = momentum;
    this->num_epochs = num_epochs;
    this->goal_error = goal_error;
    this->min_gradient = min_gradient;
    this->mode = mode;
    this->num_aux = num_aux;
    this->token = -1;
}

int HMLPL::init(){
    if(create_architecture() != SUCCESS_CREATE_ARCHITECTURE)
    {
        cerr << "ERROR: in creating architecture..." << endl;
        return ERR_INIT;
    }

    cout <<  endl;
    cout << "MAIN NET CREATED SUCCESSFULLY..." << endl;
    cout <<  endl;

    if(this->num_aux){
        if(create_aux() != SUCCESS_CREATE_ARCHITECTURE)
        {
            cerr << "ERROR: in creating aux architecture..." << endl;
            return ERR_INIT;
        }

        cout <<  endl << "AUX NETS CREATED SUCCESSFULLY..." << endl << endl;
    }


    return SUCCESS_INIT;
}

int HMLPL::create_architecture()
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

int HMLPL::initialize_weights()
{
    srand(time(0));
    if(this->lay == NULL)
    {
        cerr << "ERROR: No layers in Neural Network" << endl;
        return ERR_INIT_WEIGHTS;
    }

    cout << "Initializing weights" << endl;

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
            this->num_bias++;
        }


    }


    for (size_t j=0; j < this->num_neurons[this->num_layers-1]; j++)
    {
        this->lay[this->num_layers-1].neu[j].bias = q_set( (((double)rand())/((double)RAND_MAX))*2.0 - 1.0,
                                                            (((double)rand())/((double)RAND_MAX))*2.0 - 1.0,
                                                            (((double)rand())/((double)RAND_MAX))*2.0 - 1.0,
                                                            (((double)rand())/((double)RAND_MAX))*2.0 - 1.0);
        this->num_bias++;
    }



    return SUCCESS_INIT_WEIGHTS;
}

//==============================================================================
// INPUT & OUTPUT
//==================================================================================

int HMLPL::get_training_data(const std::string inputs_file, const std::string targets_file){

    if(this->get_inputs(inputs_file)){
        cerr << "ERROR: problem with inputs reading..." << endl;
        return ERR_INIT_DATA;
    } else cout << "Inputs injected succesfully..." << endl;

    if(this->get_targets(targets_file)){
        cerr << "ERROR: problem with targets reading..." << endl;
        return ERR_INIT_DATA;
    } else cout << "Targets injected succesfully..." << endl;

    // normalize_training_vectors();

    return SUCCESS_INIT_DATA;
}

//Get Inputs
int HMLPL::get_inputs(const std::string inputs_file)
{
    char filename[50];
    strcpy(filename, inputs_file.c_str());
    // cout << filename << endl;

    this->inputs = readMatrixFromFile(filename);
    this->num_samples = countrows(filename);
    if(countcolumns(filename) % 4){
        cerr << "ERROR: the data format is not correct..."<< endl;
        return ERR_INIT_DATA;
    } else this->num_neurons[0] = countcolumns(filename)/4;

    return SUCCESS_INIT_DATA;
}

//Get Labels
int HMLPL::get_targets(const std::string targets_file)
{
    char filename[50];
    strcpy(filename, targets_file.c_str());
    // cout << filename << endl;

    this->targets = readMatrixFromFile(filename);
    if(countcolumns(filename) % 4){
        cerr << "ERROR: the data format is not correct..."<< endl;
        return ERR_INIT_DATA;
    } else this->num_neurons[this->num_layers-1] = countcolumns(filename)/4;

    return SUCCESS_INIT_DATA;
}

int HMLPL::get_inputs_test(const std::string inputs_file)
{
    char filename[50];
    strcpy(filename, inputs_file.c_str());
    if(inputs != nullptr) { 
        freeMatrix(inputs,num_samples);
        inputs = nullptr;
    }
    if(targets != nullptr) {
        freeMatrix(targets, num_samples);
        targets = nullptr;
    }
    this->inputs = readMatrixFromFile(filename);
    this->num_samples = countrows(filename);
    if(this->num_neurons[0] != countcolumns(filename)/4){
        cerr << "ERROR: the number of input for pattern must be: " << this->num_neurons[0] << "..." << endl;
        return ERR_INIT_DATA;
    }

    return SUCCESS_INIT_DATA;
}

//==================================================================================
// FUNCTIONS FOR TRAINING
//==================================================================================


void HMLPL::train_neural_net(void)
{
    cout << endl << "=========== T R A I N ============" << endl << endl;
    switch (this->mode)
    {
    case 0:
        train_seq_mode();
        break;
    case 1:
        train_batch_mode();
    break;

    }

    if(download_weights_bias()) {
        cout << "Weights saved . . ." << endl;
    }
    else {
        cout << "Weights not saved . . ." << endl;
    } 

    training_data_export();

}

void HMLPL::train_seq_mode(void)
{
    switch (this->num_aux)
    {
    case 0:
        cout << "========== N O R M A L - S E Q U E N T I A L ==========" << endl;
        weight_vector_init();

        for(this->actual_epoch = 0; this->actual_epoch < this->num_epochs; this->actual_epoch++)
        {
            
            cout << "Epoch = " << this->actual_epoch + 1 << " | ";
            cout << "Actual_Err = " << this->actual_error  << endl;
            // cout << "Weight_gradient = " << this->gradient << " | ";
            // cout << "TOKEN = none" << endl;
            // per stamparlo nel file error_vector.txt
            error_vector[actual_epoch][0] = actual_error;
            //cout << error_vector[actual_epoch][0] << endl;

            if(this->actual_epoch > 500 & this->actual_error < this->goal_error) break;


            weight_vector_update(this->actual_epoch);

            if(this->end_flag) break;

            this->actual_error = 0;
            for(size_t ii = 0; ii < this->num_weight*4; ii++) this->gradient_vector[ii] = 0;

            //________________L E A R N I N G  L O O P____________________
            for(size_t i = 0; i < this->num_samples; i++)
            {
                // cout << "==================" << endl;
                feed_input(i);
                forward_prop();
                compute_cost(i);
                back_prop(i);
                update_weights();
                compute_grad(i);
                this->train_iter++;
            }
            //_____________________________________________________


        }
        break;
    
    default: // Learning-on-Learning mode by default
        cout << "=========== L O N L - S E Q U E N T I A L ===========" << endl;
        weight_vector_init();

        //==============================================================
        for(size_t i = 0; i < this->num_aux; i++) { 
            aux[i].weight_vector = this->weight_vector;

        }
        //==============================================================


        for(this->actual_epoch = 0; this->actual_epoch < this->num_epochs; this->actual_epoch++)
        {

            cout << "Epoch = " << this->actual_epoch + 1 << " | ";
            cout << "Actual_Err = " << this->actual_error  << " | ";
            cout << "TOKEN = " << this->token << endl;

            error_vector[actual_epoch][0] = actual_error;

            if(this->actual_epoch > 1 & this->actual_error < this->goal_error){
                break;
            }



            if(this->token < -1){
                pthread_mutex_lock(&Wt);
                this->token--;
                pthread_mutex_unlock(&Wt);
            }
            //_____________________________________________________

            if((this->token == ((-2) - 3000)) && performance_evaluation()){ // ______ T O  F I X ______
                    this->actual_epoch = this->actual_epoch;
                    pthread_mutex_lock(&Wt);
                    this->token = 0;
                    pthread_mutex_unlock(&Wt);
            }
    /**/
            //_____________________________________________________

            weight_vector_update(this->actual_epoch);

            //________________A U X  S T A R T___________________________________
            if(this->token == 0){
                    pthread_mutex_lock(&Wt);
                    this->token = 1;
                    pthread_mutex_unlock(&Wt);

                    start_aux();
            }
            //_____________________________________________________

            //________________A U X  R E T U R N________________________________
            if(this->token == this->num_aux + 1){
                this->update_epoch = this->actual_epoch;
                aux_return();
                update_predicted_weights();
                this->launch_error = this->actual_error;


                pthread_mutex_lock(&Wt);
                this->token = -2;
                pthread_mutex_unlock(&Wt);
            }
            //_____________________________________________________

            if(this->end_flag) break;
            
            this->actual_error = 0;
            for(size_t ii = 0; ii < this->num_weight*4; ii++) this->gradient_vector[ii] = 0;

            //________________L E A R N I N G  L O O P____________________
            for(size_t i = 0; i < this->num_samples; i++)
            {
                // cout << "==================" << endl;
                feed_input(i);
                forward_prop();
                compute_cost(i);
                back_prop(i);
                update_weights();
                compute_grad(i);
                this->train_iter++;
            }
            //_____________________________________________________

        }
        break;
    }
}

void HMLPL::train_batch_mode(void)
{
    switch (this->num_aux)
    {
    case 0:    
        cout << "========== N O R M A L - B A T C H ==========" << endl;
        weight_vector_init();

        for(this->actual_epoch = 0; this->actual_epoch < this->num_epochs; this->actual_epoch++)
        {

            cout << "Epoch = " << this->actual_epoch + 1 << " | ";
            cout << "Actual_Err = " << this->actual_error  << " | ";
            cout << "Weight_gradient = " << this->gradient << " | ";
            cout << "TOKEN = none" << this->token << endl;

            if(this->actual_epoch > 500 & this->actual_error < this->goal_error) break;


            weight_vector_update(this->actual_epoch);

            if(this->end_flag) break;

            this->actual_error = 0;
            for(size_t ii = 0; ii < this->num_weight*4; ii++) this->gradient_vector[ii] = 0;

            init_corrections(); // dw and dbias setted to zero

            //________________L E A R N I N G  L O O P____________________
            for(size_t i = 0; i < this->num_samples; i++)
            {
                // cout << "==================" << endl;
                feed_input(i);
                forward_prop();
                compute_cost(i);
                back_prop_batch(i);
                this->train_iter++;
            }
            
            update_weights_batch();
            compute_grad(this->actual_epoch);
            //_____________________________________________________

        }
    break;
    default: // Learning-on-Learning mode by default
        cout << "=========== L O N L - B A T C H ===========" << endl;
        weight_vector_init();

        //==============================================================
        for(size_t i = 0; i < this->num_aux; i++)
            this->aux[i].weight_vector = this->weight_vector;
        //==============================================================


        for(this->actual_epoch = 0; this->actual_epoch < this->num_epochs; this->actual_epoch++)
        {
            cout << "Epoch = " << this->actual_epoch + 1 << " | ";
            cout << "Actual_Err = " << this->actual_error  << " | ";
            cout << "Weight_gradient = " << this->gradient << " | ";
            cout << "TOKEN = " << this->token << endl;

            if(this->actual_epoch > 500 & this->actual_error < this->goal_error){
                sleep(1);
                break;
            }



            if(this->token < -1){
                pthread_mutex_lock(&Wt);
                this->token--;
                pthread_mutex_unlock(&Wt);
            }
            //_____________________________________________________

            if((this->token == ((-2) - 3000)) && performance_evaluation()){ // ______ T O  F I X ______
                    this->actual_epoch = this->actual_epoch;
                    pthread_mutex_lock(&Wt);
                    this->token = 0;
                    pthread_mutex_unlock(&Wt);
            }
    /**/
            //_____________________________________________________

            weight_vector_update(this->actual_epoch);

            //________________A U X  S T A R T___________________________________
            if(this->token == 0){
                    pthread_mutex_lock(&Wt);
                    this->token = 1;
                    pthread_mutex_unlock(&Wt);

                    start_aux();
            }
            //_____________________________________________________

            //________________A U X  R E T U R N________________________________
            if(this->token == this->num_aux + 1){
                this->update_epoch = this->actual_epoch;
                aux_return();
                update_predicted_weights();
                this->launch_error = this->actual_error;


                pthread_mutex_lock(&Wt);
                this->token = -2;
                pthread_mutex_unlock(&Wt);
            }
            //_____________________________________________________

            if(this->end_flag) break;
            
            this->actual_error = 0;
            for(size_t ii = 0; ii < this->num_weight*4; ii++) this->gradient_vector[ii] = 0;
            init_corrections(); // dw and dbias setted to zero

            //________________L E A R N I N G  L O O P____________________
            for(size_t i = 0; i < this->num_samples; i++)
            {
                // cout << "==================" << endl;
                feed_input(i);
                forward_prop();
                compute_cost(i);
                back_prop_batch(i);
                this->train_iter++;
            }
            //_____________________________________________________
            update_weights_batch();
            compute_grad(this->actual_epoch);
        }
        break;
    }
            


}

void HMLPL::feed_input(uint32_t i)
{
    int column;
    for(size_t j = 0; j < this->num_neurons[0]; j++)
    {
        column = j * 4;
        this->lay[0].neu[j].actv = q_set(this->inputs[i][column],
                                        this->inputs[i][column + 1],
                                        this->inputs[i][column + 2],
                                        this->inputs[i][column + 3]);
        // cout << "INPUT[" << j+1 << "] = " << this->lay[0].neu[j].actv << endl;
    }
}

void HMLPL::forward_prop(void)
{
    // cout << endl << "<<<<<<F O R W A R D P R O P>>>>>>" << endl << endl;
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
        }
    }

}

// Back Propogate Error
void HMLPL::back_prop(uint32_t p)
{
    int j4;
    quaternion error, tar;
    // cout << endl << "<<<<<<B A C K P R O P>>>>>>" << endl << endl;

    for(size_t i = 1; i < this->num_layers; i++){
        for(size_t j = 0; j < this->num_neurons[i]; j++){
            this->lay[i].neu[j].dactv = q_set();
            // cout << this->lay[i].neu[j].dactv << endl;
            this->lay[i].neu[j].dz = q_set();
        }
    }

    // <<<<<<Output Layer - from here>>>>>>>>>>
    for(size_t j = 0; j < this->num_neurons[this->num_layers-1]; j++) // j = number of neurons in the output layer
    {
        // ERROR IN THE OUTPUT LAYER
        j4 = j*4;
        tar = q_set(this->targets[p][j4],this->targets[p][j4 + 1],this->targets[p][j4 + 2],this->targets[p][j4 + 3]);
        error = q_diff(tar, this->lay[this->num_layers-1].neu[j].actv);

        this->lay[this->num_layers-1].neu[j].dz = q_set(error.r * this->lay[this->num_layers-1].neu[j].actv.r * (1 - this->lay[this->num_layers-1].neu[j].actv.r),
                                                error.i * this->lay[this->num_layers-1].neu[j].actv.i * (1 - this->lay[this->num_layers-1].neu[j].actv.i),
                                                error.j * this->lay[this->num_layers-1].neu[j].actv.j * (1 - this->lay[this->num_layers-1].neu[j].actv.j),
                                                error.k * this->lay[this->num_layers-1].neu[j].actv.k * (1 - this->lay[this->num_layers-1].neu[j].actv.k));

        for(size_t k = 0; k < this->num_neurons[this->num_layers-2]; k++) // k = number of neurons in the output-1 layer
        {
            this->lay[this->num_layers-2].neu[k].dw[j] = q_prod(this->lay[this->num_layers-1].neu[j].dz,q_conj(this->lay[this->num_layers-2].neu[k].actv));
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

void HMLPL::update_weights(void)
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

void HMLPL::update_predicted_weights(void){
    int r = 0;
    for(size_t i = 0; i < this->num_layers-1; i++)
    {
        for(size_t j = 0; j < this->num_neurons[i]; j++)
        {
            for(size_t k = 0; k < this->num_neurons[i+1]; k++)
            {
                // Update Weights
                if(tmp_weight_vector[r] != NO_CHANGE){
                    this->lay[i].neu[j].out_weights[k] = q_set(tmp_weight_vector[r],
                                                                tmp_weight_vector[r + 1],
                                                                tmp_weight_vector[r + 2],
                                                                tmp_weight_vector[r + 3]);
                    // Saving the gradients of the given epoch
                    cout << "weight["<< i << "][" << j << "][" << k << "] = " << this->lay[i].neu[j].out_weights[k] << endl;
                }
                else cout << "weight["<< i << "][" << j << "][" << k << "]" << " not changed..." << endl;

                this->lay[i].neu[j].dwp[k] = q_set();
                r += 4;
            }
        }
    }
    
    //cout << endl << endl;
}

//==================================================================================
// BATCH MODE
//==================================================================================

void HMLPL::init_corrections(){
    for(size_t i = 0; i < this->num_layers-1; i++)
    {

        for(size_t j = 0; j < this->num_neurons[i]; j++)
        {
            for(size_t k = 0; k < this->num_neurons[i+1]; k++)
            {
                // Initialize Output Weights for each neuron
                this->lay[i].neu[j].dw[k] = q_set();

            }

            if(i>0)
            {
                this->lay[i].neu[j].dbias = q_set();
            }
        }


    }


    for (size_t j=0; j < this->num_neurons[this->num_layers-1]; j++)
    {
        this->lay[this->num_layers-1].neu[j].dbias = q_set();
    }
}

void HMLPL::back_prop_batch(uint32_t p){
    int j4;
    quaternion error, tar;
    // cout << endl << "<<<<<<B A C K P R O P>>>>>>" << endl << endl;

    for(size_t i = 1; i < this->num_layers; i++){
        for(size_t j = 0; j < this->num_neurons[i]; j++){
            this->lay[i].neu[j].dactv = q_set();
            // cout << this->lay[i].neu[j].dactv << endl;
        }
    }

    // <<<<<<Output Layer - from here>>>>>>>>>>
    for(size_t j = 0; j < this->num_neurons[this->num_layers-1]; j++) // j = number of neurons in the output layer
    {
        // ERROR IN THE OUTPUT LAYER
        j4 = j*4;
        tar = q_set(this->targets[p][j4],this->targets[p][j4 + 1],this->targets[p][j4 + 2],this->targets[p][j4 + 3]);
        error = q_diff(tar, this->lay[this->num_layers-1].neu[j].actv);

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

void HMLPL::update_weights_batch(void){

    // cout << endl << "<<<<<<U P D A T E>>>>>>" << endl << endl;
    quaternion correction;
    double N = 1.0/((double)this->num_samples);

    for(size_t i = 0; i < this->num_layers-1; i++)
    {
        for(size_t j = 0; j < this->num_neurons[i]; j++)
        {
            for(size_t k = 0; k < this->num_neurons[i+1]; k++)
            {
                // Update Weights
                    this->lay[i].neu[j].dw[k] = q_prod(N,this->lay[i].neu[j].dw[k]);
                    // cout << i << " " << j << " " << k << " " << this->lay[i].neu[j].dw[k] << " " << mean_dw << endl;
                    correction = q_sum(q_prod(this->alpha, this->lay[i].neu[j].dw[k]),q_prod(this->momentum, this->lay[i].neu[j].dwp[k]));
                    this->lay[i].neu[j].out_weights[k] = q_sum(this->lay[i].neu[j].out_weights[k],correction);
                    this->lay[i].neu[j].dwp[k] = correction;
            }

            // Update Bias
            this->lay[i].neu[j].dbias = q_prod(N,this->lay[i].neu[j].dbias);
            correction = q_sum(q_prod(this->alpha, this->lay[i].neu[j].dbias),q_prod(this->momentum, this->lay[i].neu[j].dbp));
            this->lay[i].neu[j].bias = q_sum(this->lay[i].neu[j].bias, correction);
            this->lay[i].neu[j].dbp = correction;
        }
    }
    // Update bias output neurons
    for(size_t j = 0; j < this->num_neurons[num_layers-1]; j++)
        {
            this->lay[num_layers-1].neu[j].dbias = q_prod(N, this->lay[num_layers-1].neu[j].dbias);
            correction = q_sum(q_prod(this->alpha, this->lay[num_layers-1].neu[j].dbias),q_prod(this->momentum, this->lay[num_layers-1].neu[j].dbp));
            this->lay[num_layers-1].neu[j].bias = q_sum(this->lay[num_layers-1].neu[j].bias, correction);
            this->lay[num_layers-1].neu[j].dbp = correction;

        }

    
}


//==========================================================================================================
// FUNCTIONS FOR PERFORMANCE EVALUATION
//==========================================================================================================

// Compute Total Cost
void HMLPL::compute_cost(uint32_t i)
{
    //cout << endl << "<<<<<<C O M P U T E C O S T>>>>>>" << endl << endl;
    quaternion tmpcost;
    quaternion target;
    quaternion tavg = q_set();
    double tcost = 0;

    for(size_t j = 0; j < this->num_neurons[this->num_layers-1]; j++)
    {
        target = q_set(this->targets[i][j*4],this->targets[i][j*4 + 1],this->targets[i][j*4 + 2],this->targets[i][j*4 + 3]);
        tmpcost = q_diff(target, this->lay[this->num_layers-1].neu[j].actv);
        this->cost[j] = q_set(tmpcost.r*tmpcost.r,
                                tmpcost.i*tmpcost.i,
                                tmpcost.j*tmpcost.j,
                                tmpcost.k*tmpcost.k);
        tavg = q_set(tavg.r + fabs(tmpcost.r),tavg.i + fabs(tmpcost.i),tavg.j + fabs(tmpcost.j),tavg.k + fabs(tmpcost.k));

        tcost = tcost + this->cost[j].r/2 + this->cost[j].i/2 + this->cost[j].j/2 + this->cost[j].k/2;
    }
    this->actual_error = ((this->actual_error * i) + tcost)/(i+1); //_________T O  F I X________
    this->component_error = q_set(this->component_error.r + tavg.r,
                                this->component_error.i + tavg.i,
                                this->component_error.j + tavg.j,
                                this->component_error.k + tavg.k);
}

void HMLPL::compute_grad(uint32_t p){
    int r = 0;
    double q = 1.00002; // weight factor for the computation of the weighted average of the gradient
    double d_sum = 0;

    pthread_mutex_lock(&Wg); // =========================Gradient mutex lock=================================================

    for(size_t i = 0; i < this->num_layers - 1;i++)
    {
        for(size_t j = 0; j < this->num_neurons[i]; j++)
        {
            for(size_t k = 0;k < this->num_neurons[i+1]; k++)
            {
                /*
                this->gradient_vector[r] = (this->gradient_vector[r] * (this->train_iter - 1) * (2 - q) + fabs(this->lay[i].neu[j].dw[k].r * this->alpha) * q )/this->train_iter;
                this->gradient_vector[r + 1] = (this->gradient_vector[r + 1] * (this->train_iter - 1) * (2 - q) + fabs(this->lay[i].neu[j].dw[k].i * this->alpha) * q )/this->train_iter;
                this->gradient_vector[r + 2] = (this->gradient_vector[r + 2] * (this->train_iter - 1) * (2 - q) + fabs(this->lay[i].neu[j].dw[k].j * this->alpha) * q )/this->train_iter;
                this->gradient_vector[r + 3] = (this->gradient_vector[r + 3] * (this->train_iter - 1) * (2 - q) + fabs(this->lay[i].neu[j].dw[k].k * this->alpha) * q )/this->train_iter;
                */

                this->gradient_vector[r] = fabs(this->lay[i].neu[j].dw[k].r * this->alpha);
                this->gradient_vector[r + 1] = fabs(this->lay[i].neu[j].dw[k].i * this->alpha);
                this->gradient_vector[r + 2] = fabs(this->lay[i].neu[j].dw[k].j * this->alpha);
                this->gradient_vector[r + 3] = fabs(this->lay[i].neu[j].dw[k].k * this->alpha);
                


                d_sum = d_sum + fabs(this->gradient_vector[r]) +
                                fabs(this->gradient_vector[r + 1]) +
                                fabs(this->gradient_vector[r + 2]) +
                                fabs(this->gradient_vector[r + 3]);

                r += 4;
            }
        }
    }

    // Average gradient
    this->gradient = d_sum/ (this->num_weight * 4);
    pthread_mutex_unlock(&Wg); // =========================Gradient mutex unlock===============================================


    if(fabs(this->gradient) < this->min_gradient and (this->actual_epoch > 100))
    {
        this->end_flag = true;
    }

    if( (this->gradient < this->min_gradient * 1000) && (this->token == -1) && (this->actual_epoch > 4000) ){

        this->token = 0;
    }

}

bool HMLPL::gradient_evaluation(uint32_t weight_Id){
    bool result = true;
    double percentage;

    if(this->mode) percentage = 140; // batch
    else percentage = 140; // sequential

    double weight_gradient = ( this->gradient_vector[weight_Id] +
                                this->gradient_vector[weight_Id + 1] +
                                this->gradient_vector[weight_Id + 2] +
                                this->gradient_vector[weight_Id + 3] ) / 4;
    cout << weight_gradient << "|" << percentage/100 * this->gradient << endl;                            

     pthread_mutex_lock(&Wg); // =========================Gradient mutex lock=================================================

    if(weight_gradient < percentage/100 * this->gradient)
        result = false;

     pthread_mutex_unlock(&Wg); // =========================Gradient mutex unlock===============================================

    return result;
}

bool HMLPL::performance_evaluation(){
    bool result = false;
    cout << this->actual_error << " | " << this->launch_error << endl;

    if(this->actual_error > this->launch_error){
        cout << "------ B A C K U P -----" <<endl;
        result = true;
        this->actual_epoch = this->update_epoch;
        weight_vector_restore();
    }
    //sleep(10);


    return result;
}
//==================================================================================
// FUNCTIONS FOR TESTING
//==================================================================================

void HMLPL::test(void)
{
    this->File.open("dati/test/output_hmlp.txt");
    if(this->File.is_open()) {
        
        for(size_t i = 0; i < this->num_samples; i++){
            feed_input_test(i);
            forward_prop_test();
        }
        this->File.close();
    }
    else {
        cerr << "FILE NON APERTO CORRETTAMENTE" << endl;
    }
    
}

void HMLPL::feed_input_test(uint32_t i)
{
    int column;
    for(size_t j = 0; j < this->num_neurons[0]; j++)
    {
        column = j * 4;
        this->lay[0].neu[j].actv = q_set(this->inputs[i][column],
                                        this->inputs[i][column + 1],
                                        this->inputs[i][column + 2],
                                        this->inputs[i][column + 3]);
        //cout << "INPUT[" << j+1 << "] = " << this->lay[0].neu[j].actv << " ";
    }
    //cout << endl;
}

void HMLPL::forward_prop_test(void)
{
    // cout << endl << "<<<<<<F O R W A R D P R O P>>>>>>" << endl << endl;
    quaternion delta_x_w;

    for(size_t i = 1; i < this->num_layers; i++)
    {
        for(size_t j = 0; j < this->num_neurons[i]; j++)
        {
            this->lay[i].neu[j].z = this->lay[i].neu[j].bias;
            // cout << "z =" << this->lay[i].neu[j].z << "\t bias =" << this->lay[i].neu[j].bias << endl;

            for(size_t k = 0; k < this->num_neurons[i-1]; k++)
            {
                delta_x_w = q_prod(this->lay[i-1].neu[k].out_weights[j],this->lay[i-1].neu[k].actv);
                // cout << this->lay[i-1].neu[k].out_weights[j] << " x " << this->lay[i-1].neu[k].actv << " = "
                // << delta_x_w << endl;
                this->lay[i].neu[j].z  = q_sum(this->lay[i].neu[j].z, delta_x_w);
            }

            this->lay[i].neu[j].actv = q_sigmoid(this->lay[i].neu[j].z);
            // if (i == this->num_layers - 1)
            //    cout << "OUTPUT[" << j+1 << "] = " << this->lay[i].neu[j].actv << endl;
        }
    }

    for(size_t j = 0; j < this->num_neurons[this->num_layers-1]; j++){
        // cout << "Output["<< j+1 <<"]= " << this->lay[this->num_layers-1].neu[j].actv << " ";

        this->File << setprecision(16) << this->lay[this->num_layers-1].neu[j].actv.r << " ";
        this->File << setprecision(16) << this->lay[this->num_layers-1].neu[j].actv.i << " ";
        this->File << setprecision(16) << this->lay[this->num_layers-1].neu[j].actv.j << " ";
        this->File << setprecision(16) << this->lay[this->num_layers-1].neu[j].actv.k << " ";
    }


    // cout << endl;

    this->File << endl;
}


//==================================================================================
// FUNCTIONS FOR PREDICTION
//==================================================================================


void HMLPL::predict(uint32_t num_step, uint32_t input_step){


    cout << endl << "=========== P R E D I C T I O N ============" << endl; 
    string filename = "dati/prediction_hmlp/output_hmlp_prediction_step_" + to_string(input_step) + ".txt";
    cout << filename << endl << endl;
    
    this->File.open(filename);
    if(this->File.is_open()) {
        for(size_t i = 0; i < num_step; i++){
            if(!(i % input_step)) feed_input_predict(i);
            forward_prop_predict();
        }
        /**/
        this->File.close();
    }
    else {
        cerr << "FILE NON APERTO CORRETTAMENTE" << endl;
    }
}
void HMLPL::feed_input_predict(uint32_t i)
{
    int column;
    for(size_t j = 0; j < this->num_neurons[0]; j++)
    {
        column = j * 4;
        this->lay[0].neu[j].actv = q_set(this->inputs[i][column],
                                        this->inputs[i][column + 1],
                                        this->inputs[i][column + 2],
                                        this->inputs[i][column + 3]);
        //cout << "INPUT[" << j+1 << "] = " << this->lay[0].neu[j].actv << " ";
        if(i == 0){
            // cout << "_________________________________________" << endl;
            this->File << setprecision(16) << this->lay[0].neu[j].actv.r << " ";
            this->File << setprecision(16) << this->lay[0].neu[j].actv.i << " ";
            this->File << setprecision(16) << this->lay[0].neu[j].actv.j << " ";
            this->File << setprecision(16) << this->lay[0].neu[j].actv.k << " ";
        }
    }
    //cout << endl;
    if(i == 0) this->File << endl;
}

void HMLPL::forward_prop_predict(void){
    // cout << endl << "<<<<<<F O R W A R D P R O P>>>>>>" << endl << endl;
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
        }
    }

    for(size_t j = 0; j < this->num_neurons[this->num_layers-1]; j++){
        this->lay[0].neu[j].actv = q_set(this->lay[this->num_layers-1].neu[j].actv.r,
                                        this->lay[this->num_layers-1].neu[j].actv.i,
                                        this->lay[this->num_layers-1].neu[j].actv.j,
                                        this->lay[this->num_layers-1].neu[j].actv.k); // *********************************************

        this->File << setprecision(16) << this->lay[this->num_layers-1].neu[j].actv.r << " ";
        this->File << setprecision(16) << this->lay[this->num_layers-1].neu[j].actv.i << " ";
        this->File << setprecision(16) << this->lay[this->num_layers-1].neu[j].actv.j << " ";
        this->File << setprecision(16) << this->lay[this->num_layers-1].neu[j].actv.k << " ";
    }
    this->File << endl;
}

//==================================================================================
// WEIGHT VECTOR MANAGING
//==================================================================================

// Weight vector init
void HMLPL::weight_vector_init(){

    this->weight_vector = new double*[this->num_weight * 4]();
    for(size_t i = 0; i < this->num_weight * 4; i++){

        this->weight_vector[i] = new double[this->num_epochs]();
    }

    this->tmp_weight_vector = new double[this->num_weight * 4]();

    this->gradient_vector = new double[this->num_weight * 4]();

    error_vector = new double*[num_epochs]();
    for(size_t i = 0; i < num_epochs; i++){

        error_vector[i] = new double[0]();
    }
}

void HMLPL::weight_vector_update(uint32_t actual_epoch){
    int r = 0;
    for(size_t l = 0; l < this->num_layers-1; l++)
        for(size_t n = 0; n < this->num_neurons[l]; n++)
            for(size_t w = 0; w < this->num_neurons[l+1]; w++){
                    this->weight_vector[r][actual_epoch] = this->lay[l].neu[n].out_weights[w].r;
                    this->weight_vector[r + 1][actual_epoch] = this->lay[l].neu[n].out_weights[w].i;
                    this->weight_vector[r + 2][actual_epoch] = this->lay[l].neu[n].out_weights[w].j;
                    this->weight_vector[r + 3][actual_epoch] = this->lay[l].neu[n].out_weights[w].k;
                    r += 4;
            }
}


void HMLPL::export_weight_vector(){
    ofstream f;
    f.open("dati/test/weight_vector.txt");

    for(size_t i = 0; i < num_epochs; i++) {
        for(size_t j = 0; j < num_weight * 4; j++) {
            f << weight_vector[j][i] << " ";
        }
        f << endl;
    }    
    f.close();

    f.open("dati/test/error_vector.txt");
    for(size_t i = 0; i < num_epochs; i++) {
        f << error_vector[i][0] << endl;
    }    
    f.close();
}

void HMLPL::weight_vector_restore(void){
    int r = 0;
    for(size_t l = 0; l < this->num_layers-1; l++)
        for(size_t n = 0; n < this->num_neurons[l]; n++)
            for(size_t w = 0; w < this->num_neurons[l+1]; w++){
                    this->lay[l].neu[n].out_weights[w].r = this->weight_vector[r][this->update_epoch];
                    this->lay[l].neu[n].out_weights[w].i = this->weight_vector[r + 1][this->update_epoch];
                    this->lay[l].neu[n].out_weights[w].j = this->weight_vector[r + 2][this->update_epoch];
                    this->lay[l].neu[n].out_weights[w].k = this->weight_vector[r + 3][this->update_epoch];

                    this->lay[l].neu[n].dwp[w] = q_set();

                    r += 4;
            }

}

bool HMLPL::download_weights_bias() {
    ofstream f, fb;
    f.open("dati/weights.txt");

    if(f.is_open()){
        for(size_t l = 0; l < this->num_layers-1; l++)
            for(size_t n = 0; n < this->num_neurons[l]; n++)
                for(size_t w = 0; w < this->num_neurons[l+1]; w++){
                        f << setprecision(16) << this->lay[l].neu[n].out_weights[w].r << endl;
                        f << setprecision(16) << this->lay[l].neu[n].out_weights[w].i << endl;
                        f << setprecision(16) << this->lay[l].neu[n].out_weights[w].j << endl;
                        f << setprecision(16) << this->lay[l].neu[n].out_weights[w].k << endl;
                } 
                
        f.close();
    }

    fb.open("dati/bias.txt");
    if(fb.is_open()) {
        for(size_t l = 0; l < this->num_layers; l++)
            for(size_t n = 0; n < this->num_neurons[l]; n++){
                        fb << setprecision(16) << this->lay[l].neu[n].bias.r << endl;
                        fb << setprecision(16) << this->lay[l].neu[n].bias.i << endl;
                        fb << setprecision(16) << this->lay[l].neu[n].bias.j << endl;
                        fb << setprecision(16) << this->lay[l].neu[n].bias.k << endl;
                } 
        
        fb.close();
    }
    else {
        return false;
    }

    return true;
}

bool HMLPL::upload_weights_bias(const std::string weights_file, const std::string bias_file) {

    cout << "UPLOAD" << endl;
    char filename[50];
    double bArr[this->num_bias * 4];
    double wArr[this->num_weight * 4];
    int i = 0;
    string tp;
    ifstream f, fb;

    strcpy(filename, weights_file.c_str());
    f.open(filename);

    if (f.is_open()){   //checking whether the file is open
        i = 0;
        
        while(getline(f, tp)){ //read data from file object and put it into string.
            wArr[i] = atof(tp.c_str());
            i++;
        }
        if(i != this->num_weight * 4) {
            cerr << "The number of read weights " << i << " does not match the number of required weights " << this->num_weight << endl;
            return false;
        }
        else {
            i = 0;
            for(size_t l = 0; l < this->num_layers-1; l++)
                for(size_t n = 0; n < this->num_neurons[l]; n++)
                    for(size_t w = 0; w < this->num_neurons[l+1]; w++) {
                            this->lay[l].neu[n].out_weights[w] = q_set(wArr[i], wArr[i + 1], wArr[i + 2], wArr[i + 3]);
                            cout << i << ") w[" << l << "][" << n << "]["<< w << "]:" << this->lay[l].neu[n].out_weights[w] << endl;
                            i += 4;
                        }
        }

        f.close(); //close the file object.
    }
    else {
        cerr << "Error: impossible to open the file..." << endl;
        return false;
    }

    strcpy(filename, bias_file.c_str());
    fb.open(filename);

    if (fb.is_open()){   //checking whether the file is open
        i = 0;

        cout << this->num_bias * 4 << endl;
    
        
        while(getline(fb, tp)){ //read data from file object and put it into string.
            bArr[i] = atof(tp.c_str());
            i++;
        }

        if(i != this->num_bias * 4) {
            cerr << "The number of read bias " << i << " does not match the number of required bias " << this->num_bias << endl;
            return false;
        }
        else {
            i = 0;
            
            for(size_t l = 0; l < this->num_layers; l++)
                for(size_t n = 0; n < this->num_neurons[l]; n++) {
                        this->lay[l].neu[n].bias = q_set(bArr[i], bArr[i + 1], bArr[i + 2], bArr[i + 3]);
                        cout << i << ") w[" << l << "][" << n << "]:" << this->lay[l].neu[n].bias << endl;
                        i += 4;
                    }
        }
        
        fb.close(); //close the file object.
    }
    else {
        cerr << "Error: impossible to open the file..." << endl;
        return false;
    }
    return true;
}

//==================================================================================
// AUX MANAGING
//==================================================================================

/**
 * @brief 
 * 
 * 
 * @return int 
 */
int HMLPL::create_aux()
{
    uint32_t num_aux_layers = 3;
    uint32_t num_hidden_neurons[num_aux_layers - 2] = {3};
    double aux_alpha = 0.001;
    double aux_momentum = 0.05;
    uint32_t num_epoch_aux = 10000;
    int res;
    uint32_t r = 0;
    uint32_t s = 0;
    uint32_t first_weight, last_weight;

    uint32_t tmp = 0;
    uint32_t step = num_weight / num_aux;
    uint32_t q = num_weight % num_aux;

    aux = new HMLP[num_aux]();

    // pthread_create(...,...,...,void *arg) arguments vector
    if(num_aux && arg == nullptr) arg = new t_arg[num_aux]();

    for(size_t i = 0; i < num_aux; i++)
    {
        aux[i].set_HMLP(i,num_layers,num_hidden_neurons,aux_alpha,aux_momentum,num_epoch_aux,0);
        aux[i].num_neurons[0] = 1;
        aux[i].num_neurons[2] = 1;
        aux[i].start_epoch = 1;
        aux[i].step = 1;

        res = aux[i].init();
        if(res){
            cerr << "ERROR: in initializating aux#" << i+1 << endl;
            return res;
        }



        first_weight = tmp;
        if(q > 0) {
            last_weight = tmp + step;
            tmp = tmp + step + 1;
            q--;
        }
        else {
            last_weight = tmp + step - 1;
            tmp = tmp + step;
        }
        aux[i].num_work = last_weight - first_weight + 1;
        //cout << "Num ele: "<< last_weight<< " " << first_weight<< " " << aux[i].num_work << endl;
        aux[i].work_vector = new u_int[aux[i].num_work]();

        // cout << "La rete aux "<< i+1 << "prende indici da " << aux[i].first_weight + 1 << " a " << aux[i].last_weight + 1 << endl;
    }

    for(size_t i = 0; i < num_weight; i++){
        aux[r].work_vector[s] = i;
        r++;
        if(r == num_aux){
            r = 0;
            s++;
        }
    }

    return res;
}

void HMLPL::start_aux(void){

    int res;
    launch_epoch = actual_epoch - 1;


    for(size_t i = 0; i < num_aux ; i++)
    {
        arg[i].aux_Id = i;

        //cout << "LANCIANDO AUX#" << (arg[c].aux_Id) << endl;
        res = pthread_create(&(aux[i].tid), NULL, aux_thread, &arg[i]);
        if (res != 0) {
            printf("Thread creation failed\n");
            exit(EXIT_FAILURE);
        }
    }
    // char g; cin >> g;

}

void HMLPL::aux_return(void){
    int res;
    int numthread;

    // printf("Sono nella aux_return\n");
    for(size_t i = 0; i < num_aux ; i++)
    {

        res = pthread_join(this->aux[i].tid, (void **)&numthread);
        if (res){
            printf("pthread_join failed\n");
            exit(EXIT_FAILURE);
        }
        else {
            printf("%d) Thread concluso %d\n", i, numthread);
        }
    }
    // for(int i = 0; i<num_weights;i++) printf("%d) %f \n",i,tmp_weight_vector[i]);

}

void *aux_thread(void *arg) {

    double tau = 1.4;
    int r = 0;
    int indx = 0;
    quaternion tmp;

    HMLPL *main = main1;

    t_arg *p = (t_arg *)arg;
    indx = (p->aux_Id);
    HMLP *aux = &(main->aux[indx]);
    //cout << "H E L L O " << indx << endl;

    cout << "Thread#" << indx << " is working . . ." << endl;

    for(size_t i = 0; i < aux->num_work; i++)
    {
        // cout << "Thread#" << indx << " is processing weight " << aux->work_vector[i] << ". . ." << endl;
        r = aux->work_vector[i] * 4;
        if(main->gradient_evaluation(r)) // The weights are computed by auxs
        {

            aux->reset_net(main->launch_epoch);
            aux->train_neural_net(aux->work_vector[i]);
            tmp = aux->predict_weight(main->launch_epoch * tau);
            main->tmp_weight_vector[r] = tmp.r;
            main->tmp_weight_vector[r + 1] = tmp.i;
            main->tmp_weight_vector[r + 2] = tmp.j;
            main->tmp_weight_vector[r + 3] = tmp.k;
            /**/
        }
        else // The weights are not changed
        {
            main->tmp_weight_vector[r] = NO_CHANGE;
        }
    }



    pthread_mutex_lock(&Wt);
    main->token++;
    pthread_mutex_unlock(&Wt);




    pthread_exit((void *)indx);
    // pthread_exit(nullptr);
}
/**/