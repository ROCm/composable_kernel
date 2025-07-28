#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wzero-as-null-pointer-constant"
#include "rapidjson/writer.h"
#include "rapidjson/stringbuffer.h"
#include "rapidjson/document.h"
#include "rapidjson/rapidjson.h"
#pragma GCC diagnostic pop

#define START_JSON_DUMP_FILE(file_name)                                             \
    std::ofstream file(file_name);                                                  \
    if(!file.is_open())                                                             \
    {                                                                               \
        throw std::runtime_error("Could not open file: " + std::string(file_name)); \
    }                                                                               \
    rapidjson::StringBuffer s;                                                      \
    rapidjson::Writer<rapidjson::StringBuffer> writer(s);                           \
    writer.StartObject();

#define END_JSON_DUMP_FILE() \
    writer.EndObject();      \
    file << s.GetString();   \
    file.close();

#define ADD_KEY_VALUE(key, value) add_key_value_pair(writer, key, value);
#define ADD_PERF_TO_JSON(_time, gflops, gbytes) add_perf_to_json(writer, _time, gflops, gbytes);

template <typename T>
void add_key_value_pair(rapidjson::Writer<rapidjson::StringBuffer>& writer,
                        const char* key,
                        T value)
{
    writer.Key(key);
    if constexpr(std::is_same<T, const char*>::value)
    {
        writer.String(value, static_cast<rapidjson::SizeType>(std::strlen(value)));
    }
    else if constexpr(std::is_same<T, std::string>::value)
    {
        writer.String(value.c_str(), static_cast<rapidjson::SizeType>(value.length()));
    }
    else if constexpr(std::is_floating_point<T>::value)
    {
        writer.Double(static_cast<double>(value));
    }
    else if constexpr(std::is_integral<T>::value)
    {
        writer.Int64(static_cast<int64_t>(value));
    }
    else
    {
        static_assert(std::is_same<T, const char*>::value || std::is_floating_point<T>::value ||
                          std::is_integral<T>::value,
                      "Unsupported type for JSON serialization");
    }
}

static void add_perf_to_json(rapidjson::Writer<rapidjson::StringBuffer>& writer,
                             float time,
                             float gflops,
                             float gbytes)
{
    std::string roster("perf");
    writer.String(roster.c_str(), static_cast<rapidjson::SizeType>(roster.length()));

    writer.StartArray();
    writer.StartObject();

    add_key_value_pair(writer, "time", time);
    add_key_value_pair(writer, "gflops", gflops);
    add_key_value_pair(writer, "gbytes", gbytes);

    writer.EndObject();
    writer.EndArray();
}

template <typename ALayout,
          typename BLayout,
          typename CLayout,
          typename ADataType,
          typename BDataType,
          typename CDataType,
          typename GemmConfig,
          template <typename>
          typename DTypeTraits>
void dump_gemm_json_results(const std::string& json_filename,
                            int M,
                            int N,
                            int K,
                            int stride_A,
                            int stride_B,
                            int stride_C,
                            bool persistent,
                            bool pass,
                            float ave_time,
                            float tflops,
                            float gb_per_sec,
                            const std::string& kernel_name = "gemm_basic")
{
    START_JSON_DUMP_FILE(json_filename);
    ADD_KEY_VALUE("name", kernel_name);
    ADD_KEY_VALUE("M", M);
    ADD_KEY_VALUE("N", N);
    ADD_KEY_VALUE("K", K);
    ADD_KEY_VALUE("stride_A", stride_A);
    ADD_KEY_VALUE("stride_B", stride_B);
    ADD_KEY_VALUE("stride_C", stride_C);
    ADD_KEY_VALUE("A_layout", ALayout::name);
    ADD_KEY_VALUE("B_layout", BLayout::name);
    ADD_KEY_VALUE("C_layout", CLayout::name);
    using TraitsADataType = DTypeTraits<ADataType>;
    using TraitsBDataType = DTypeTraits<BDataType>;
    using TraitsCDataType = DTypeTraits<CDataType>;
    ADD_KEY_VALUE("A_type", TraitsADataType::name);
    ADD_KEY_VALUE("B_type", TraitsBDataType::name);
    ADD_KEY_VALUE("C_type", TraitsCDataType::name);
    ADD_KEY_VALUE("structured_sparsity", GemmConfig::UseStructuredSparsity ? "on" : "off");
    ADD_KEY_VALUE("persistent", persistent ? "on" : "off");
    ADD_KEY_VALUE("verification", pass ? "pass" : "fail");
    ADD_PERF_TO_JSON(ave_time, tflops, gb_per_sec);
    END_JSON_DUMP_FILE();
}

void dump_batched_gemm_json_results(const std::string& json_filename,
                                    const std::string& op_name,
                                    int M,
                                    int N,
                                    int K,
                                    int stride_A,
                                    int stride_B,
                                    int stride_C,
                                    int batch_stride_A,
                                    int batch_stride_B,
                                    int batch_stride_C,
                                    int batch_count,
                                    bool pass,
                                    float ave_time,
                                    float tflops,
                                    float gb_per_sec,
                                    const std::string& kernel_name = "batched_gemm_basic")
{
    START_JSON_DUMP_FILE(json_filename);
    ADD_KEY_VALUE("name", kernel_name);
    ADD_KEY_VALUE("op_name", op_name);
    ADD_KEY_VALUE("M", M);
    ADD_KEY_VALUE("N", N);
    ADD_KEY_VALUE("K", K);
    ADD_KEY_VALUE("stride_A", stride_A);
    ADD_KEY_VALUE("stride_B", stride_B);
    ADD_KEY_VALUE("stride_C", stride_C);
    ADD_KEY_VALUE("batch_stride_A", batch_stride_A);
    ADD_KEY_VALUE("batch_stride_B", batch_stride_B);
    ADD_KEY_VALUE("batch_stride_C", batch_stride_C);
    ADD_KEY_VALUE("batch_count", batch_count);
    ADD_KEY_VALUE("verification", pass ? "pass" : "fail");
    ADD_PERF_TO_JSON(ave_time, tflops, gb_per_sec)
    END_JSON_DUMP_FILE();
}

template <typename ALayout, typename BLayout, typename CLayout>
void dump_grouped_gemm_json_results(const std::string& json_filename,
                                    const std::string& op_name,
                                    int group_count,
                                    bool pass,
                                    float ave_time,
                                    float tflops,
                                    float gb_per_sec,
                                    const std::string& kernel_name = "grouped_gemm")
{
    START_JSON_DUMP_FILE(json_filename);
    ADD_KEY_VALUE("name", kernel_name);
    ADD_KEY_VALUE("op_name", op_name);
    ADD_KEY_VALUE("group_count", group_count);
    ADD_KEY_VALUE("A_layout", ALayout::name);
    ADD_KEY_VALUE("B_layout", BLayout::name);
    ADD_KEY_VALUE("C_layout", CLayout::name);
    ADD_KEY_VALUE("verification", pass ? "pass" : "fail");
    ADD_PERF_TO_JSON(ave_time, tflops, gb_per_sec)
    END_JSON_DUMP_FILE();
}
