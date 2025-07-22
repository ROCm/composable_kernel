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
