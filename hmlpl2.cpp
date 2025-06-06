#include"hmlpl.h"

using namespace std;

//==================================================================================
// FUNCTIONS FOR PREDICTION
//==================================================================================


void HMLPL::predict_2(uint32_t num_step, uint32_t input_step) {


    cout << endl << "=========== P R E D I C T I O N ============" << endl; 
    string filename = "dati/prediction_hmlp/output_hmlp_prediction_step_" + to_string(num_step) + ".txt";
    cout << filename << endl << endl;

    bool input_Injection = false;

    if(this->num_neurons[this->num_layers - 1] == 1) {

        // Create outputVector
        uint32_t size = this->num_neurons[0] + num_step;
        double** outputVector = new double*[size]();
        for(size_t i = 0; i < size; i++) {
            outputVector[i] = new double[this->num_neurons[this->num_layers - 1] * 4]();
        }
        
        this->File.open(filename);
        if(this->File.is_open()) {
            for(size_t i = 0; i < num_step; i++){
                if(!(i % input_step)) input_Injection = true;
                feed_input_predict(outputVector,i,input_Injection);
                forward_prop_predict(outputVector,i);
                input_Injection = false;
            }
            this->File.close();
        }
        else {
            cerr << "FILE NON APERTO CORRETTAMENTE" << endl;
        }
            
        
        
        // Destroy outputVector
        for(size_t i = 0; i < size; i++) {
            delete[] outputVector[i];
        }
        delete outputVector;        
        this->File.close();
    }
    else {
        cout << "There are more than 1 output neurons. This function requires a network with just one output neuron." << endl;
    }
}

void HMLPL::feed_input_predict(double** output_vector, uint32_t i, bool input_Injection) {
    uint32_t k = 0;
    uint32_t c = this->num_neurons[0]/this->num_neurons[this->num_layers - 1];

    cout << i << "]";
    if (input_Injection) {

        for (size_t iii = 0; iii < this->num_neurons[0] ; iii++) {
            for (size_t j = 0; j < 4 ; j++) {
                for(size_t k = 0; k < this->num_neurons[this->num_layers - 1]; k++) {
                    output_vector[i + iii][j + (k * 4)] = 
                        this->inputs[i][(iii * 4) + j];
                }
            }

        }
        cout << "] Injected";
    }
    cout << endl;
    
    for(size_t j = 0; j < this->num_neurons[0]; j++)
        {
            this->lay[0].neu[j].actv = q_set(output_vector[i + j][0 + (k*4)],
                                            output_vector[i + j][1 + (k*4)],
                                            output_vector[i + j][2 + (k*4)],
                                            output_vector[i + j][3 + (k*4)]);
            cout << "INPUT[" << j+1 << "][" <<  k+1 <<  "] = " << this->lay[0].neu[j].actv << " " << endl;
            
            if((j+1) % c == 0) k++;            
    }

}

void HMLPL::forward_prop_predict(double** output_vector, uint32_t i) {
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
    
    uint32_t row = i + this->num_neurons[0]; // riga dell'output_vector da stampare;

    for(size_t j = 0; j < this->num_neurons[this->num_layers-1]; j++) {
        output_vector[row][j*4 + 0] = this->lay[this->num_layers-1].neu[j].actv.r;
        output_vector[row][j*4 + 1] = this->lay[this->num_layers-1].neu[j].actv.i;
        output_vector[row][j*4 + 2] = this->lay[this->num_layers-1].neu[j].actv.j;
        output_vector[row][j*4 + 3] = this->lay[this->num_layers-1].neu[j].actv.k;

        cout << "OUTPUT[" << j+1 << "] = " << this->lay[this->num_layers-1].neu[j].actv << "\t";

        this->File << setprecision(16) << this->lay[this->num_layers-1].neu[j].actv.r << " ";
        this->File << setprecision(16) << this->lay[this->num_layers-1].neu[j].actv.i << " ";
        this->File << setprecision(16) << this->lay[this->num_layers-1].neu[j].actv.j << " ";
        this->File << setprecision(16) << this->lay[this->num_layers-1].neu[j].actv.k << " ";
    }
    cout << endl << "----------------------------------------------------" << endl;
    this->File << endl;
}

int HMLPL::training_data_export(void){
    string filename = "dati/training_statistics.txt";
    reportFile.open(filename,std::ios_base::app);

    if(reportFile.is_open()){
        time_t s;
        struct tm* current_time;
        // time in seconds
        s = time(NULL);
        // to get current time
        current_time = localtime(&s);

        reportFile << actual_epoch << " " << actual_error << endl;

        reportFile.close();
        return EXIT_SUCCESS;
    }
    else {
        cerr << "ERROR: file " << filename << " not opened correctly . . ." << endl;
        return EXIT_FAILURE;
    }    
}