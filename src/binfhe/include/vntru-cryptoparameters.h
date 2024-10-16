// 
#ifndef _VNTRU_CRYPTOPARAMETERS_H_
#define _VNTRU_CRYPTOPARAMETERS_H_

#include "lattice/lat-hal.h"
#include "math/discretegaussiangenerator.h"
#include "math/nbtheory.h"
#include "utils/serializable.h"
#include "utils/utilities.h"

#include "binfhe-constants.h"

#include "lwe-ciphertext.h"
#include "lwe-keyswitchkey.h"
#include "lwe-cryptoparameters.h"

#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <map>

namespace lbcrypto {
    class VectorNTRUCryptoParams : public Serializable {
private:
    // modulus for the RingGSW/RingLWE scheme
    NativeInteger m_Q{};
    // modulus for the RingLWE scheme
    NativeInteger m_q{};
    // ring dimension for RingGSW/RingLWE scheme
    uint32_t m_N{};

    uint32_t m_baseG{};

    uint32_t m_baseR{};

    uint32_t m_digitsG{};

    std::vector<NativeInteger> m_digitsR;

    std::vector<NativeInteger> m_Gpower;

    std::vector<int32_t> m_logGen;

    DiscreteGaussianGeneratorImpl<NativeVector> m_dgg;

    std::map<uint32_t, std::vector<NativeInteger>> m_Gpower_map;

    std::shared_ptr<ILNativeParams> m_polyParams;

    std::vector<NativeInteger> m_gateConst;

    std::vector<NativePoly> m_monomials;

    BINFHE_METHOD m_method{BINFHE_METHOD::INVALID_METHOD};

    SecretKeyDist m_keyDist{SecretKeyDist::UNIFORM_TERNARY};

    uint32_t m_numAutoKeys{};
public:
    VectorNTRUCryptoParams() = default;
     explicit VectorNTRUCryptoParams(uint32_t N, NativeInteger Q, NativeInteger q, uint32_t baseG, uint32_t baseR,
                                 BINFHE_METHOD method, double std, SecretKeyDist keyDist = UNIFORM_TERNARY,
                                 bool signEval = false, uint32_t numAutoKeys = 10)
        : m_Q(Q),
          m_q(q),
          m_N(N),
          m_baseG(baseG),
          m_baseR(baseR),
          m_polyParams{std::make_shared<ILNativeParams>(2 * N, Q)},
          m_method(method),
          m_keyDist(keyDist),
          m_numAutoKeys(numAutoKeys) {
        if (!IsPowerOfTwo(baseG))
            OPENFHE_THROW(config_error, "Gadget base should be a power of two.");
        if ((method == LMKCDEY) & (numAutoKeys == 0))
            OPENFHE_THROW(config_error, "numAutoKeys should be greater than 0.");
        auto logQ{log(m_Q.ConvertToDouble())};
        m_digitsG = static_cast<uint32_t>(std::ceil(logQ / log(static_cast<double>(m_baseG))));
        m_dgg.SetStd(std);
        PreCompute(signEval);
    }
    void PreCompute(bool signEval = false);

    uint32_t GetN() const {
        return m_N;
    }

    const NativeInteger& GetQ() const {
        return m_Q;
    }

    const NativeInteger& Getq() const {
        return m_q;
    }

    uint32_t GetBaseG() const {
        return m_baseG;
    }

    uint32_t GetDigitsG() const {
        return m_digitsG;
    }

    uint32_t GetBaseR() const {
        return m_baseR;
    }

    uint32_t GetNumAutoKeys() const {
        return m_numAutoKeys;
    }

    const std::vector<NativeInteger>& GetDigitsR() const {
        return m_digitsR;
    }

    const std::shared_ptr<ILNativeParams> GetPolyParams() const {
        return m_polyParams;
    }

    const std::vector<NativeInteger>& GetGPower() const {
        return m_Gpower;
    }

    const std::vector<int32_t>& GetLogGen() const {
        return m_logGen;
    }

    const std::map<uint32_t, std::vector<NativeInteger>>& GetGPowerMap() const {
        return m_Gpower_map;
    }

    const DiscreteGaussianGeneratorImpl<NativeVector>& GetDgg() const {
        return m_dgg;
    }

    const std::vector<NativeInteger>& GetGateConst() const {
        return m_gateConst;
    }

    const NativePoly& GetMonomial(uint32_t i) const {
        return m_monomials[i];
    }

    BINFHE_METHOD GetMethod() const {
        return m_method;
    }

    SecretKeyDist GetKeyDist() const {
        return m_keyDist;
    }

    bool operator==(const VectorNTRUCryptoParams& other) const {
        return m_N == other.m_N && m_Q == other.m_Q && m_baseR == other.m_baseR && m_baseG == other.m_baseG;
    }

    bool operator!=(const VectorNTRUCryptoParams& other) const {
        return !(*this == other);
    }

    
    template <class Archive>
    void save(Archive& ar, std::uint32_t const version) const {
        ar(::cereal::make_nvp("bN", m_N));
        ar(::cereal::make_nvp("bQ", m_Q));
        ar(::cereal::make_nvp("bq", m_q));
        ar(::cereal::make_nvp("bR", m_baseR));
        ar(::cereal::make_nvp("bG", m_baseG));
        ar(::cereal::make_nvp("bmethod", m_method));
        ar(::cereal::make_nvp("bs", m_dgg.GetStd()));
        ar(::cereal::make_nvp("bdigitsG", m_digitsG));
        ar(::cereal::make_nvp("bparams", m_polyParams));
        ar(::cereal::make_nvp("numAutoKeys", m_numAutoKeys));
    }

    template <class Archive>
    void load(Archive& ar, std::uint32_t const version) {
        if (version > SerializedVersion()) {
            OPENFHE_THROW(deserialize_error, "serialized object version " + std::to_string(version) +
                                                 " is from a later version of the library");
        }
        ar(::cereal::make_nvp("bN", m_N));
        ar(::cereal::make_nvp("bQ", m_Q));
        ar(::cereal::make_nvp("bq", m_q));
        ar(::cereal::make_nvp("bR", m_baseR));
        ar(::cereal::make_nvp("bG", m_baseG));
        ar(::cereal::make_nvp("bmethod", m_method));
        double sigma = 0;
        ar(::cereal::make_nvp("bs", sigma));
        m_dgg.SetStd(sigma);
        ar(::cereal::make_nvp("bdigitsG", m_digitsG));
        ar(::cereal::make_nvp("bparams", m_polyParams));
        ar(::cereal::make_nvp("numAutoKeys", m_numAutoKeys));

        PreCompute();
    }
    
    std::string SerializedObjectName() const override {
        return "VectorNTRUCryptoParams";
    }
    static uint32_t SerializedVersion() {
        return 1;
    }

    void Change_BaseG(uint32_t BaseG) {
        if (m_baseG != BaseG) {
            m_baseG  = BaseG;
            m_Gpower = m_Gpower_map[m_baseG];
            m_digitsG =
                static_cast<uint32_t>(std::ceil(log(m_Q.ConvertToDouble()) / log(static_cast<double>(m_baseG))));
        }
    }


    };
}

#endif