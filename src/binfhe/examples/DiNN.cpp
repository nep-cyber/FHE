#include <stdio.h>
#include <iostream>
#include <iomanip>
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <string>
#include <vector>
#include <sys/time.h>
#include <sys/wait.h>
#include <unistd.h>
#include "binfhecontext.h"


#define PrintRES false
#define SET DiNN

// Defines
#define VERBOSE 1
#define STATISTICS true
#define WRITELATEX true
#define N_PROC 3

// #define CARD_TESTSET 10000
#define CARD_TESTSET 10000


// The expected topology of the provided neural network is 256:30:10
#define NUM_NEURONS_LAYERS 3
#define NUM_NEURONS_INPUT  256
#define NUM_NEURONS_HIDDEN 30
#define NUM_NEURONS_OUTPUT 10



// Files are expected in the executable's directory
// #define PATH_TO_FILES       "buildotests/test/" //TODO FIXME!
#define FILE_TXT_IMG        "../../../../MINIST/img_test.txt"
#define FILE_TXT_BIASES     "../../../../MINIST/biases.txt"
#define FILE_TXT_WEIGHTS    "../../../../MINIST/weights.txt"
#define FILE_TXT_LABELS     "../../../../MINIST/labels.txt"

#define FILE_LATEX          "results_LaTeX.tex"
#define FILE_STATISTICS     "results_stats.txt"

// Tweak neural network
#define THRESHOLD_WEIGHTS  9
#define THRESHOLD_SCORE -100

#define MSG_SLOTS    440
#define TORUS_SLOTS  200
using namespace lbcrypto;
using namespace std;


void deleteTensor(int*** tensor, int dim_mat, const int* dim_vec);
void deleteMatrix(int**  matrix, int dim_mat);





int main()
{
    // Input data
    const int n_images = CARD_TESTSET;    // 10000

     // Network specific
    const int num_wire_layers = NUM_NEURONS_LAYERS - 1;   // 3-1=2
    const int num_neuron_layers = NUM_NEURONS_LAYERS;     //3
    const int num_neurons_in = NUM_NEURONS_INPUT;         // 256
    const int num_neurons_hidden = NUM_NEURONS_HIDDEN;    // 30
    const int num_neurons_out = NUM_NEURONS_OUTPUT;       // 10

    // Vector of number of neurons in layer_in, layer_H1, layer_H2, ..., layer_Hd, layer_out;
    const int topology[num_neuron_layers] = {num_neurons_in, num_neurons_hidden, num_neurons_out};  // 256,30,10


    const int space_msg = MSG_SLOTS;                



    const bool clamp_biases  = false;  
    const bool clamp_weights = false;

    const bool statistics        = STATISTICS;   //true
    const bool writeLaTeX_result = WRITELATEX;   //true

    const int threshold_biases  = THRESHOLD_WEIGHTS;    // 
    const int threshold_weights = THRESHOLD_WEIGHTS;    // 
    const int threshold_scores  = THRESHOLD_SCORE;      // 


    const int total_num_hidden_neurons = n_images * NUM_NEURONS_HIDDEN;  //
    const double avg_bs  = 1./NUM_NEURONS_HIDDEN;     // 1/30
    const double avg_total_bs  = 1./total_num_hidden_neurons; 
    const double avg_img = 1./n_images;               
    const double clocks2seconds = 1. / CLOCKS_PER_SEC;
    const int slice = (n_images+N_PROC-1)/N_PROC;    

    // Huge arrays
    int*** weights = new int**[num_wire_layers];  // allocate and fill matrices holding the weights   // {0，1}*in*out
    int ** biases  = new int* [num_wire_layers];  // allocate and fill vectors holding the biases    // 
    int ** images  = new int* [n_images];
    int  * labels  = new int  [n_images];

    // Temporary variables
    string line;
    int el, l;
    int num_neurons_current_layer_in, num_neurons_current_layer_out;






    auto cc = BinFHEContext();
    cc.GenerateBinFHEContext(SET, XZNEW);
    LWEPrivateKey sk = cc.KeyGen();
    cc.NBTKeyGen(sk);
    const LWEPrivateKey in_out_params = sk;
    const auto& LWEParams = cc.GetParams()->GetLWEParams();
    const auto LWEscheme = cc.GetLWEScheme();
    const int qKS = LWEParams->GetqKS().ConvertToInt<uint32_t>(); 
    // cout<<"qKS = "<<qKS<<endl;




    if (VERBOSE) cout << "IMPORT PIXELS, WEIGHTS, BIASES, and LABELS FROM FILES" << endl;
    if (VERBOSE) cout << "Reading images (regardless of dimension) from " << FILE_TXT_IMG << endl;
    ifstream file_images(FILE_TXT_IMG);

    for (int img=0; img<n_images; ++img)  // 10000
        images[img] = new int[num_neurons_in];
    
    // cout<<"end img"<<endl;
    int filling_image = 0;
    int image_count = 0;

    // while(getline(file_images, line))
    for (int img=0; img<n_images*num_neurons_in; ++img)
    {
        getline(file_images, line);
        // cout<<"line="<<line<<endl;
        images[filling_image][image_count++] = stoi(line);         // 10000 * 256
        if (image_count == num_neurons_in)
        {
            image_count = 0;
            filling_image++;
        }
    }
    file_images.close();


    if (VERBOSE) cout << "Reading weights from " << FILE_TXT_WEIGHTS << endl;
    ifstream file_weights(FILE_TXT_WEIGHTS);

    num_neurons_current_layer_out = topology[0];
    for (l=0; l<num_wire_layers; ++l)
    {
        num_neurons_current_layer_in = num_neurons_current_layer_out;
        num_neurons_current_layer_out = topology[l+1];

        weights[l] = new int*[num_neurons_current_layer_in];
        for (int i = 0; i<num_neurons_current_layer_in; ++i)
        {
            weights[l][i] = new int[num_neurons_current_layer_out];
            for (int j=0; j<num_neurons_current_layer_out; ++j)
            {
                getline(file_weights, line);
                el = stoi(line);
                if (clamp_weights)
                {
                    if (el < -threshold_weights)
                        el = -threshold_weights;
                    else if (el > threshold_weights)
                        el = threshold_weights;
                }
                weights[l][i][j] = el;
            }
        }
    }
    file_weights.close();
    
    if (VERBOSE) cout << "Reading biases from " << FILE_TXT_BIASES << endl;
    ifstream file_biases(FILE_TXT_BIASES);

    num_neurons_current_layer_out = topology[0];
    for (l=0; l<num_wire_layers; ++l)
    {
        num_neurons_current_layer_in = num_neurons_current_layer_out;
        num_neurons_current_layer_out = topology[l+1];

        biases [l] = new int [num_neurons_current_layer_out];  // 0 256   1 30
        for (int j=0; j<num_neurons_current_layer_out; ++j)
        {
            getline(file_biases, line);
            el = stoi(line);
            if (clamp_biases)
            {
                if (el < -threshold_biases)
                    el = -threshold_biases;
                else if (el > threshold_biases)
                    el = threshold_biases;
                // else, nothing as it holds that: -threshold_biases < el < threshold_biases
            }
            biases[l][j] = el;
        }
    }
    file_biases.close();


    if (VERBOSE) cout << "Reading labels from " << FILE_TXT_LABELS << endl;
    ifstream file_labels(FILE_TXT_LABELS);
    for (int img=0; img<n_images; ++img)
    {
        getline(file_labels, line);
        labels[img] = stoi(line);
    }
    file_labels.close();

    if (VERBOSE) cout << "Import done. END OF IMPORT" << endl;



    // // Temporary variables and Pointers to existing arrays for convenience
    bool notSameSign;
    // // Torus32 mu, phase;

    int** weight_layer;
    int * bias;
    int * image;
    int label;
    int pixel;
    int x, w, w0;

    // // LweSample *multi_sum, *enc_image, *bootstrapped;
    // LWECiphertext *enc_image;

    
    std::vector<LWECiphertext> enc_image(topology[0]);
    std::vector<LWECiphertext> multi_sum(topology[1]);
    std::vector<LWECiphertext> bootstrapped(topology[1]);
    std::vector<LWECiphertext> multi_sum2(topology[2]);


    // cout<<"enc_image.size()="<<enc_image.size()<<endl;


    int multi_sum_clear[num_neurons_hidden];
    int output_clear   [num_neurons_out];

    int max_score = 0;
    int max_score_clear = 0;
    int class_enc = 0;
    int class_clear = 0;
    int score = 0;
    int score_clear = 0;


    bool failed_bs = false;
    // // Counters
    int count_errors = 0;
    int count_errors_with_failed_bs = 0;
    int count_disagreements = 0;
    int count_disagreements_with_failed_bs = 0;
    int count_disag_pro_clear = 0;
    int count_disag_pro_hom = 0;
    int count_wrong_bs = 0;

    int r_count_errors, r_count_disagreements, r_count_disag_pro_clear, r_count_disag_pro_hom, r_count_wrong_bs, r_count_errors_with_failed_bs, r_count_disagreements_with_failed_bs;
    double r_total_time_network, r_total_time_bootstrappings;

    // // For statistics output
    double avg_time_per_classification = 0.0;
    double avg_time_per_bootstrapping = 0.0;
    double total_time_bootstrappings = 0.0;
    double total_time_network = 0.0;
    double error_rel_percent = 0.0;

    // // Timings
    clock_t net_begin,bs_begin,bs_end,net_end;
    double time_bootstrappings,time_per_bootstrapping,time_per_classification;


    pid_t pids[N_PROC];
    int pipes[N_PROC][2];

    for (int id_proc=0; id_proc < N_PROC; ++id_proc)
    {
        
        pipe(pipes[id_proc]);  
        pid_t pid = fork();
        if (pid != 0)
        {
            pids [id_proc] = pid;
            close(pipes[id_proc][1]);
        }else{
            close(pipes[id_proc][0]);
            for (int img = id_proc*slice; img < ( (id_proc+1)*slice) && (img< n_images); /*img*/ ){    //slice = 10000/4 = 2500
                image = images[img];
                label = labels[img];
                ++img;


                num_neurons_current_layer_out= topology[0];
                num_neurons_current_layer_in = num_neurons_current_layer_out;



                for (int i = 0; i < num_neurons_current_layer_in; ++i)  // 
                {
                    pixel = image[i];   // 1 or -1
                    int mu = pixel<0? (pixel+space_msg):pixel;
                    enc_image[i] = cc.Encrypt(sk, mu,BOOTSTRAPPED,space_msg,LWEParams->GetqKS());

                }
                // ========  Layer 1  ========  y_0 = w_0*x+b_0 
                net_begin = clock();
                for (l=0; l<num_wire_layers - 1 ; ++l)     // 
                {
                    
                    num_neurons_current_layer_in = num_neurons_current_layer_out; //256
                    num_neurons_current_layer_out= topology[l+1]; // 30
                    bias = biases[l];            // 30
                    weight_layer = weights[l];   // 256*30
                    for (int j=0; j<num_neurons_current_layer_out; ++j) // 30
                    {
                        w0 = bias[j];
                        multi_sum_clear[j] = w0;
                        int mu = w0<0? (w0+space_msg):w0;
                        mu = mu*qKS/space_msg; 
 
                        /**----------------noise free Enc(bias) = (b,0)---------------------**/
                        NativeVector a_zero(LWEParams->Getn(),LWEParams->GetqKS());
                        NativeInteger m_mu{mu};
                        multi_sum[j] = std::make_shared<LWECiphertextImpl>(std::move(a_zero),std::move(m_mu));//mu = b
                        multi_sum[j]->SetptModulus(NativeInteger(space_msg));

                        LWEPlaintext result3;


                        for (int i=0; i<num_neurons_current_layer_in; ++i)//256
                        {
                            x = image [i];  // 
                            w = weight_layer[i][j];  // 
                            multi_sum_clear[j] += x * w; // 

                            LWECiphertext ctprep = std::make_shared<LWECiphertextImpl>(*enc_image[i]);
                            /**----------------Enc(x_i) = w_i*Enc(x_i)---------------------**/
                            LWEscheme->EvalMultConstEq(ctprep,NativeInteger(w));
                            LWEscheme->EvalAddEq(multi_sum[j],ctprep); //
                        }
                        cc.Decrypt(sk, multi_sum[j], &result3,space_msg);//
                    }
                }//end Layer

                bs_begin = clock();
                for (int j=0; j<num_neurons_current_layer_out; ++j)//30
                {
                    LWECiphertext ctMS = LWEscheme->ModSwitch(LWEParams->Getq(), multi_sum[j]);
                    bootstrapped[j] = cc.Sign_Bootstrap(ctMS);// qKS
                }
                bs_end = clock();
                time_bootstrappings = bs_end - bs_begin;
                total_time_bootstrappings += time_bootstrappings;
                time_per_bootstrapping = time_bootstrappings*avg_bs;
                if (VERBOSE) cout <<  time_per_bootstrapping*clocks2seconds << " [sec/bootstrapping]" << endl;


                failed_bs = false;
                for (int j=0; j<num_neurons_current_layer_out; ++j)//30
                {
                    LWEPlaintext result;
                    cc.Decrypt(sk, bootstrapped[j], &result,space_msg);//
                    int sign_result = result>space_msg/2?-1:1;
                    notSameSign = multi_sum_clear[j]*sign_result < 0; // 
                    if (notSameSign)
                    {
                        count_wrong_bs++;
                        failed_bs = true;
                    }
                }
                

                 // ========  Layer 2  ========
                max_score = threshold_scores;
                max_score_clear = threshold_scores;


                bias = biases[l];
                weight_layer = weights[l];
                l++;
                num_neurons_current_layer_in = num_neurons_current_layer_out;  //30
                num_neurons_current_layer_out= topology[l]; // l == L = 2     //10

                for (int j=0; j<num_neurons_current_layer_out; ++j)//10
                {
                    w0 = bias[j];
                    output_clear[j] = w0;
                    int mu = w0<0? (w0+space_msg):w0;
                    mu = mu*qKS/space_msg;  
                    /**----------------noise free Enc(bias) = (b,0)---------------------**/
                    NativeVector a_zero(LWEParams->Getn(),LWEParams->GetqKS());
                    NativeInteger m_mu{mu};
                    multi_sum2[j] = std::make_shared<LWECiphertextImpl>(std::move(a_zero),std::move(m_mu));//mu = b
                    multi_sum2[j]->SetptModulus(NativeInteger(space_msg));


                    for (int i=0; i<num_neurons_current_layer_in; ++i)  //30
                    {
                        w = weight_layer[i][j];

                        LWECiphertext ctprep = std::make_shared<LWECiphertextImpl>(*bootstrapped[i]);
                        LWEscheme->EvalMultConstEq(ctprep,NativeInteger(w));
                        LWEscheme->EvalAddEq(multi_sum2[j],ctprep); 
                        if (multi_sum_clear[i] < 0)
                            output_clear[j] -= w;
                        else
                            output_clear[j] += w;
                    }
                    LWEPlaintext result;
                    cc.Decrypt(sk, multi_sum2[j], &result,space_msg);
                    score = result>space_msg/2?result-space_msg:result;

                    if (score > max_score)
                    {
                        max_score = score; //
                        class_enc = j;
                    }

                    score_clear = output_clear[j];
                    if (score_clear > max_score_clear)
                    {
                        max_score_clear = score_clear;
                        class_clear = j;
                    }

                }
                if(PrintRES){
                cout<<"class_enc = "<<class_enc<<endl;
                cout<<"class_clear = "<<class_clear<<endl;   //
                cout<<"label = "<<label<<endl;
                }

                if (class_enc != label)
                {
                    count_errors++;
                    if (failed_bs)
                        count_errors_with_failed_bs++;
                }

                if (class_clear != class_enc)
                {
                    count_disagreements++;   //
                    if (failed_bs)
                        count_disagreements_with_failed_bs++;

                    if (class_clear == label)  //
                        count_disag_pro_clear++;
                    else if (class_enc == label) //
                        count_disag_pro_hom++;
                }
                net_end = clock();
                time_per_classification = net_end - net_begin;
                total_time_network += time_per_classification;
                if (VERBOSE&&PrintRES) cout << "            "<< time_per_classification*clocks2seconds <<" [sec/classification]" << endl;



            }//end for 10000/4 = 2500




            FILE* stream = fdopen(pipes[id_proc][1], "w");
            fprintf(stream, "%d,%d,%d,%d,%d,%d,%d,%lf,%lf\n", count_errors, count_disagreements, count_disag_pro_clear, count_disag_pro_hom, count_wrong_bs,
                    count_errors_with_failed_bs, count_disagreements_with_failed_bs, total_time_network, total_time_bootstrappings);
            fclose(stream);
            exit(0);
        }//end else   
    }//end id_proc

    for (auto pid : pids)
    {
        waitpid(pid, 0, 0);
    }

    time_per_classification = 0.0;
    time_per_bootstrapping = 0.0;
    for (int id_proc=0; id_proc<N_PROC; ++id_proc)
    {
        FILE* stream = fdopen(pipes[id_proc][0], "r");
        fscanf(stream, "%d,%d,%d,%d,%d,%d,%d,%lf,%lf\n", &r_count_errors, &r_count_disagreements,
               &r_count_disag_pro_clear, &r_count_disag_pro_hom, &r_count_wrong_bs, &r_count_errors_with_failed_bs,
               &r_count_disagreements_with_failed_bs, &r_total_time_network, &r_total_time_bootstrappings);
        fclose(stream);
        count_errors += r_count_errors;
        count_disagreements += r_count_disagreements;
        count_disag_pro_clear += r_count_disag_pro_clear;
        count_disag_pro_hom += r_count_disag_pro_hom;
        count_wrong_bs += r_count_wrong_bs;
        count_errors_with_failed_bs += r_count_errors_with_failed_bs;
        count_disagreements_with_failed_bs += r_count_disagreements_with_failed_bs;
        time_per_classification += r_total_time_network;
        time_per_bootstrapping += r_total_time_bootstrappings;
    }

    if (statistics)
    {
        ofstream of(FILE_STATISTICS);
        // Print some statistics
        error_rel_percent = count_errors*avg_img*100;
        //Avg. time for the evaluation of the network (seconds):
        avg_time_per_classification = time_per_classification*avg_img*clocks2seconds;
        avg_time_per_bootstrapping  = time_per_bootstrapping *avg_total_bs *clocks2seconds;

        cout << "Errors: " << count_errors << " / " << n_images << " (" << error_rel_percent << " %)" << endl;
        cout << "Disagreements: " << count_disagreements;  //clear
        cout << " (pro-clear/pro-hom: " << count_disag_pro_clear << " / " << count_disag_pro_hom << ")" << endl;
        cout << "Wrong bootstrappings: " << count_wrong_bs << endl;
        cout << "Errors with failed bootstrapping: " << count_errors_with_failed_bs << endl;
        cout << "Disagreements with failed bootstrapping: " << count_disagreements_with_failed_bs << endl;
        cout << "Avg. time for the evaluation of the network (seconds): " << avg_time_per_classification << endl;
        cout << "Avg. time per bootstrapping (seconds): " << avg_time_per_bootstrapping << endl;

        of << "Errors: " << count_errors << " / " << n_images << " (" << error_rel_percent << " %)" << endl;
        of << "Disagreements: " << count_disagreements;
        of << " (pro-clear/pro-hom: " << count_disag_pro_clear << " / " << count_disag_pro_hom << ")" << endl;
        of << "Wrong bootstrappings: " << count_wrong_bs << endl;
        of << "Errors with failed bootstrapping: " << count_errors_with_failed_bs << endl;
        of << "Disagreements with failed bootstrapping: " << count_disagreements_with_failed_bs << endl;
        of << "Avg. time for the evaluation of the network (seconds): " << avg_time_per_classification << endl;
        of << "Avg. time per bootstrapping (seconds): " << avg_time_per_bootstrapping << endl;

        // Write some statistics
        cout << "\n Wrote statistics to file: " << FILE_STATISTICS << endl << endl;
        of.close();
    }

    if (writeLaTeX_result)
    {
        cout << "\n Wrote LaTeX_result to file: " << FILE_LATEX << endl << endl;
        ofstream of(FILE_LATEX);
        of << "%\\input{"<<FILE_LATEX<<"}" << endl;

        of << "% Experiments detailed" << endl;
        of << "\\newcommand{\\EXPnumBS}{$"<<total_num_hidden_neurons<<"$}" << endl;
        of << "\\newcommand{\\EXPbsEXACT}{$"    <<avg_time_per_bootstrapping<<"$\\ [sec/bootstrapping]}" << endl;
        of << "\\newcommand{\\EXPtimeEXACT}{$"  <<avg_time_per_classification<<"$\\ [sec/classification]}" << endl;

        of << "\\newcommand{\\EXPnumERRabs}{$"  <<count_errors<<"$}" << endl;
        of << "\\newcommand{\\EXPnumERRper}{$"  <<error_rel_percent<<"\\ \\%$}" << endl;
        of << "\\newcommand{\\EXPwrongBSabs}{$" <<count_wrong_bs<<"$}" << endl;
        of << "\\newcommand{\\EXPwrongDISabs}{$"<<count_disagreements_with_failed_bs<<"$}" << endl;
        of << "\\newcommand{\\EXPdis}{$"        <<count_disagreements<<"$}" << endl;
        of << "\\newcommand{\\EXPclear}{$"      <<count_disag_pro_clear<<"$}" << endl;
        of << "\\newcommand{\\EXPhom}{$"        <<count_disag_pro_hom<<"$}" << endl << endl;

        of << "\\begin{Verbatim}[frame=single,numbers=left,commandchars=+\\[\\]%" << endl;
        of << "]" << endl;
        of << "### Classified samples: +EXPtestset" << endl;
        of << "Time per bootstrapping: +EXPbsEXACT" << endl;
        of << "Errors: +EXPnumERRabs / +EXPtestset (+EXPnumERRper)" << endl;
        of << "Disagreements: +EXPdis" << endl;
        of << "(pro-clear/pro-hom: +EXPclear / +EXPhom)" << endl;
        of << "Wrong bootstrappings: +EXPwrongBSabs" << endl;
        of << "Disagreements with wrong bootstrapping: +EXPwrongDISabs" << endl;
        of << "Avg. time for the evaluation of the network: +EXPtimeEXACT" << endl;
        of << "\\end{Verbatim}" << endl;
        of.close();
    }

    // free memory
    // delete_gate_bootstrapping_secret_keyset(secret);
    // delete_gate_bootstrapping_parameters(params);

    deleteTensor(weights,num_wire_layers, topology);
    deleteMatrix(biases, num_wire_layers);
    deleteMatrix(images, n_images);
    delete[] labels;

    return 0;

}

void deleteMatrix(int** matrix, int dim_mat)
{
    for (int i=0; i<dim_mat; ++i)
    {
        delete[] matrix[i];
    }
    delete[] matrix;
}

void deleteTensor(int*** tensor, int dim_tensor, const int* dim_vec)
{
    int** matrix;
    int dim_mat;
    for (int i=0; i<dim_tensor; ++i)
    {
        matrix =  tensor[i];
        dim_mat = dim_vec[i];
        deleteMatrix(matrix, dim_mat);
    }
    delete[] tensor;
}