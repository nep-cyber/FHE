//  
#include "vntru-cryptoparameters.h"


namespace lbcrypto {

//TODO XZDDF
void VectorNTRUCryptoParams::PreCompute(bool signEval) {
    // Computes baseR^i (only for AP bootstrapping)
    if (m_method == BINFHE_METHOD::AP) {
        auto&& logq = log(m_q.ConvertToDouble());
        auto digitCountR{static_cast<size_t>(std::ceil(logq / log(static_cast<double>(m_baseR))))};
        m_digitsR.clear();
        m_digitsR.reserve(digitCountR);
        BasicInteger value{1};
        for (size_t i = 0; i < digitCountR; ++i, value *= m_baseR)
            m_digitsR.emplace_back(value);
    }

    // Computes baseG^i
    if (signEval) {
        constexpr uint32_t baseGlist[]  = {1 << 14, 1 << 18, 1 << 27};
        constexpr double logbaseGlist[] = {noexcept(log(1 << 14)), noexcept(log(1 << 18)), noexcept(log(1 << 27))};
        constexpr NativeInteger nativebaseGlist[] = {1 << 14, 1 << 18, 1 << 27};
        auto logQ{log(m_Q.ConvertToDouble())};
        for (size_t j = 0; j < 3; ++j) {
            NativeInteger vTemp{1};
            auto tempdigits{static_cast<size_t>(std::ceil(logQ / logbaseGlist[j]))};
            std::vector<NativeInteger> tempvec(tempdigits);
            for (size_t i = 0; i < tempdigits; ++i) {
                tempvec[i] = vTemp;
                vTemp      = vTemp.ModMulFast(nativebaseGlist[j], m_Q);
            }
            if (m_baseG == baseGlist[j])
                m_Gpower = tempvec;
            m_Gpower_map[baseGlist[j]] = std::move(tempvec);
        }
    }
    else {
        m_Gpower.reserve(m_digitsG);
        NativeInteger vTemp{1};
        for (uint32_t i = 0; i < m_digitsG; ++i) {
            m_Gpower.push_back(vTemp);
            vTemp = vTemp.ModMulFast(NativeInteger(m_baseG), m_Q);
        }
    }

    // Sets the gate constants for supported binary operations
    m_gateConst = {
        NativeInteger(5) * (m_q >> 3),   // OR
        NativeInteger(7) * (m_q >> 3),   // AND
        NativeInteger(1) * (m_q >> 3),   // NOR
        NativeInteger(3) * (m_q >> 3),   // NAND
        NativeInteger(5) * (m_q >> 3),   // XOR_FAST
        NativeInteger(1) * (m_q >> 3),   // XNOR_FAST
        NativeInteger(7) * (m_q >> 3),   // MAJORITY
        NativeInteger(11) * (m_q / 12),  // AND3
        NativeInteger(7) * (m_q / 12),   // OR3
        NativeInteger(15) * (m_q >> 4),  // AND4
        NativeInteger(9) * (m_q >> 4)    // OR4
    };

    // Computes polynomials X^m - 1 that are needed in the accumulator for the
    // CGGI bootstrapping
    if (m_method == BINFHE_METHOD::GINX) {
        constexpr NativeInteger one{1};
        m_monomials.reserve(2 * m_N);
        for (uint32_t i = 0; i < m_N; ++i) {
            NativePoly aPoly(m_polyParams, Format::COEFFICIENT, true);
            aPoly[0].ModSubFastEq(one, m_Q);  // -1
            aPoly[i].ModAddFastEq(one, m_Q);  // X^m
            aPoly.SetFormat(Format::EVALUATION);
            m_monomials.push_back(std::move(aPoly));
        }
        for (uint32_t i = 0; i < m_N; ++i) {
            NativePoly aPoly(m_polyParams, Format::COEFFICIENT, true);
            aPoly[0].ModSubFastEq(one, m_Q);  // -1
            aPoly[i].ModSubFastEq(one, m_Q);  // -X^m
            aPoly.SetFormat(Format::EVALUATION);
            m_monomials.push_back(std::move(aPoly));
        }
    }

    if (m_method == LMKCDEY || m_method == XZNEW) {
        constexpr uint32_t gen{5};
        m_logGen.clear();
        uint32_t M{2 * m_N};
        m_logGen.resize(M);
        uint32_t gPow{1};
        m_logGen[M - gPow] = M;  // for -1    //m_logGen[15] = 16
        for (uint32_t i = 1; i < m_N/2; ++i) {// 1~3
            gPow               = (gPow * gen) % M;
            m_logGen[gPow]     = i;   // gPow     = 5,9,13,1
            m_logGen[M - gPow] = -i;  // M - gPow = 13,7,3,15
        }
    }
}

};  // namespace lbcrypto
