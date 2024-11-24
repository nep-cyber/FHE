
#include "binfhecontext.h"

using namespace lbcrypto;

int main() {
    // Sample Program: Step 1: Set CryptoContext
    
   // cc.GenerateBinFHEContext(P128G, XZDDF);
    int right_times=0;
    int cyc_times=100;
    int m0=1;
    int m1=1;
    LWEPlaintext result;
    // Sample Program: Step 2: Key Generation
    for(int i=0;i<cyc_times;i++)
    {
        auto cc = BinFHEContext();
        cc.GenerateBinFHEContext(N128G, XZNEW);
        // cc.GenerateBinFHEContext(P128G, XZDDF);
        auto sk = cc.KeyGen();
        cc.NBTKeyGen(sk);
        auto ct1 = cc.Encrypt(sk, m0);
        auto ct2 = cc.Encrypt(sk, m1);
        LWECiphertext ctAND1;
        clock_t start = clock();
        ctAND1 = cc.EvalBinGate(NAND, ct1, ct2);
        std::cout << "Bootstrapping in " << float(clock()-start)*1000/CLOCKS_PER_SEC<<"ms" << std::endl;
        cc.Decrypt(sk, ctAND1, &result);
        if(result == (1- m0*m1))
        {
            right_times++;
            // cout<<right_times<<" right"<<endl;
            cout<<" right"<<endl;
        }else{
            cout<<" false"<<endl;
            // cout<<right_times<<" false"<<endl;
        }
    }
    std::cout << "right_times = "<<right_times << std::endl;
    std::cout << "Accuracy = "<<double(right_times)/double(cyc_times)*100 << "%"<< std::endl;



    std::cout << "Result of encrypted computation of ( "<<m0<<" NAND "<<m1<<" ) = " << result << std::endl;

    return 0;
}
