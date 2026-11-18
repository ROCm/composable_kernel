// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#pragma once

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <functional>
#include <limits>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>

namespace ck_tile::test {
namespace reference {

// Strides and offsets count float elements, not bytes. Inputs are decoded values,
// before external descales; a sequence can point into a batch or a ragged allocation.
struct FloatView
{
    const float* data       = nullptr;
    std::size_t size        = 0;
    std::size_t offset      = 0;
    std::size_t head_stride = 0;
    std::size_t row_stride  = 0;
    std::size_t dim_stride  = 1;

    float operator()(std::size_t head, std::size_t row, std::size_t dim = 0) const
    {
        return data[offset + head * head_stride + row * row_stride + dim * dim_stride];
    }
};

struct Sequence
{
    FloatView q, k, v;
    std::size_t query_length    = 0;
    std::size_t key_length      = 0;
    std::size_t query_heads     = 1;
    std::size_t kv_heads        = 1;
    std::size_t query_dimension = 0;
    std::size_t value_dimension = 0;
    float score_scale           = 1;
    float q_descale             = 1;
    float k_descale             = 1;
    float v_descale             = 1;

    // An empty predicate means dense attention. Reuse CK's actual mask with
    // [mask](auto q, auto k) { return mask.IsOutOfSinkBound(q, k); }.
    std::function<bool(std::size_t, std::size_t)> is_masked;

    // Empty means all rows. Explicit rows are checked for every query head;
    // returned row identities make sampled coverage distinguishable from full.
    std::vector<std::size_t> sampled_query_rows;

    // Optional per-query-head logits for one extra softmax term with V=0.
    // These logits are already scaled; -infinity is equivalent to no sink.
    std::vector<float> sink_logits;
};

struct ReferenceRow
{
    std::size_t sequence;
    std::size_t head;
    std::size_t query;
    std::size_t valid_keys;
    std::vector<float> output;
    float lse;
    // Independent, unquantized P: A[d] = sum_k P[k] * abs(V[k,d]).
    std::vector<double> absolute_contribution;
    std::vector<double> maximum_value;
    double score_roundoff = 0;
};

struct Reference
{
    std::vector<ReferenceRow> rows;
    std::size_t key_chunk_size = 128;
};

struct Output
{
    FloatView o;
    std::optional<FloatView> lse;
};

struct Thresholds
{
    double max_abs_error   = 0.015625;
    double min_row_cosine  = 0.99998;
    double mean_row_cosine = 0.99999;
    double lse_atol        = 0.02;
    double lse_rtol        = 0.002;
};

// A tested P/O roundoff contract, not a universal hardware error guarantee.
// See accuracy_contract.md for the FP32 budget assumptions and fault controls.
struct AccuracyContract
{
    int mantissa_bits;
    std::size_t key_tile_size;

    double UnitRoundoff() const { return std::ldexp(1.0, -mantissa_bits - 1); }
    bool RequiresStrict() const { return mantissa_bits == 10; }
    // FP16 half-subnormal spacing; BF16 permits flushing subnormals to zero.
    double AbsoluteRoundoff() const { return std::ldexp(1.0, mantissa_bits == 10 ? -25 : -126); }
};

struct Metrics
{
    std::size_t checked_rows             = 0;
    std::size_t checked_elements         = 0;
    std::size_t scanned_output_elements  = 0;
    std::size_t scanned_lse_rows         = 0;
    std::size_t checked_lse_rows         = 0;
    std::size_t nonfinite_output_count   = 0;
    std::size_t invalid_lse_count        = 0;
    std::size_t lse_mismatch_count       = 0;
    double max_abs_error                 = 0;
    double max_relative_error            = 0;
    double rms_error                     = 0;
    double relative_rms_error            = 0;
    double min_row_cosine                = 1;
    double mean_row_cosine               = 1;
    double max_lse_abs_error             = 0;
    double max_lse_relative_error        = 0;
    double max_component_ratio           = 0;
    double max_row_relative_l2           = 0;
    double max_row_norm_drift            = 0;
    double max_head_gain                 = 0;
    std::size_t component_mismatch_count = 0;
    std::size_t component_mismatch_rows  = 0;
    bool strict_passed                   = false;
    bool contract_passed                 = false;
    bool passed                          = false;
};

namespace detail {

inline std::size_t AddExtent(std::size_t offset, std::size_t length, std::size_t stride)
{
    if(length > 0 && stride > (std::numeric_limits<std::size_t>::max() - offset) / length)
        throw std::invalid_argument("attention view extent overflows");
    return offset + length * stride;
}

inline void
ValidateView(const FloatView& view, std::size_t heads, std::size_t rows, std::size_t dimensions)
{
    if(heads == 0 || rows == 0 || dimensions == 0)
        return;
    auto last = AddExtent(view.offset, heads - 1, view.head_stride);
    last      = AddExtent(last, rows - 1, view.row_stride);
    last      = AddExtent(last, dimensions - 1, view.dim_stride);
    if(view.data == nullptr || last >= view.size)
        throw std::invalid_argument("attention view exceeds its allocation");
}

inline void ValidateSequence(const Sequence& sequence)
{
    if(sequence.query_heads == 0 || sequence.kv_heads == 0 ||
       sequence.query_heads % sequence.kv_heads != 0 || sequence.query_dimension == 0 ||
       sequence.value_dimension == 0)
        throw std::invalid_argument("invalid attention dimensions or GQA head ratio");
    if(!std::isfinite(sequence.score_scale) || !std::isfinite(sequence.q_descale) ||
       !std::isfinite(sequence.k_descale) || !std::isfinite(sequence.v_descale))
        throw std::invalid_argument("nonfinite attention scale");
    if(!sequence.sink_logits.empty() && sequence.sink_logits.size() != sequence.query_heads)
        throw std::invalid_argument("attention sink head count mismatch");
    for(const auto sink : sequence.sink_logits)
        if(!std::isfinite(sink) && sink != -std::numeric_limits<float>::infinity())
            throw std::invalid_argument("invalid attention sink logit");
    ValidateView(sequence.q, sequence.query_heads, sequence.query_length, sequence.query_dimension);
    ValidateView(sequence.k, sequence.kv_heads, sequence.key_length, sequence.query_dimension);
    ValidateView(sequence.v, sequence.kv_heads, sequence.key_length, sequence.value_dimension);
    auto sorted_rows = sequence.sampled_query_rows;
    std::sort(sorted_rows.begin(), sorted_rows.end());
    if(!sorted_rows.empty() &&
       (sorted_rows.back() >= sequence.query_length ||
        std::adjacent_find(sorted_rows.begin(), sorted_rows.end()) != sorted_rows.end()))
        throw std::invalid_argument("sample rows are duplicated or out of range");
}

inline void ValidateFiniteInput(const FloatView& view,
                                std::size_t heads,
                                std::size_t rows,
                                std::size_t dimensions)
{
    for(std::size_t head = 0; head < heads; ++head)
        for(std::size_t row = 0; row < rows; ++row)
            for(std::size_t dim = 0; dim < dimensions; ++dim)
                if(!std::isfinite(view(head, row, dim)))
                    throw std::invalid_argument("nonfinite decoded attention input");
}

inline std::size_t SelectedRows(const Sequence& sequence)
{
    return sequence.sampled_query_rows.empty() ? sequence.query_length
                                               : sequence.sampled_query_rows.size();
}

inline std::size_t QueryRow(const Sequence& sequence, std::size_t selected_row)
{
    return sequence.sampled_query_rows.empty() ? selected_row
                                               : sequence.sampled_query_rows[selected_row];
}

inline bool IsMasked(const Sequence& sequence, std::size_t query, std::size_t key)
{
    return sequence.is_masked && sequence.is_masked(query, key);
}

inline bool IsEmptyRow(const Sequence& sequence, std::size_t query, std::size_t head)
{
    if(!sequence.sink_logits.empty() && std::isfinite(sequence.sink_logits[head]))
        return false;
    for(std::size_t key = 0; key < sequence.key_length; ++key)
        if(!IsMasked(sequence, query, key))
            return false;
    return true;
}

inline double RelativeError(double absolute_error, double reference_magnitude)
{
    return reference_magnitude != 0 ? absolute_error / reference_magnitude
           : absolute_error == 0    ? 0
                                    : std::numeric_limits<double>::infinity();
}

inline double Gamma(double operations)
{
    constexpr double unit = 0x1p-24;
    if(operations < 0 || operations * unit >= 1)
        throw std::invalid_argument("attention FP32 budget exceeds its supported operation count");
    return operations * unit / (1 - operations * unit);
}

inline double ComponentLimit(const ReferenceRow& row,
                             std::size_t dim,
                             std::size_t key_length,
                             std::size_t reference_chunk_size,
                             const AccuracyContract& contract)
{
    constexpr double unit = 0x1p-24;
    const double device_tiles =
        key_length / contract.key_tile_size + (key_length % contract.key_tile_size != 0);
    const double reference_tiles =
        key_length / reference_chunk_size + (key_length % reference_chunk_size != 0);
    // Both the device and the independent FP32 oracle accumulate. The exp
    // allowance is an explicit test policy assumption, not an ISA theorem.
    const double beta = 2 * Gamma(static_cast<double>(row.valid_keys)) +
                        Gamma(3 * device_tiles + 8) + Gamma(3 * reference_tiles + 8) +
                        std::expm1(4 * row.score_roundoff) + 4 * unit;
    const double target        = std::abs(static_cast<double>(row.output[dim]));
    const double contribution  = row.absolute_contribution[dim];
    const double u             = contract.UnitRoundoff();
    const double floor         = contract.AbsoluteRoundoff();
    const double before_output = beta * contribution + Gamma(2) * target + u * contribution +
                                 floor * row.valid_keys * row.maximum_value[dim];
    // Retain the cross term from rounding O after the P/FP32 error.
    return (1 + u) * before_output + u * target + floor;
}

} // namespace detail

// The score chunk and FP32 accumulators use O(key_chunk_size + Dq + Dv) scratch.
// Only selected O rows are retained. No quantization of P occurs in this oracle.
inline Reference Compute(const std::vector<Sequence>& sequences, std::size_t key_chunk_size = 128)
{
    if(key_chunk_size == 0)
        throw std::invalid_argument("attention key chunk must be nonzero");
    Reference reference;
    reference.key_chunk_size      = key_chunk_size;
    const float negative_infinity = -std::numeric_limits<float>::infinity();
    for(std::size_t batch = 0; batch < sequences.size(); ++batch)
    {
        const auto& sequence = sequences[batch];
        detail::ValidateSequence(sequence);
        detail::ValidateFiniteInput(
            sequence.q, sequence.query_heads, sequence.query_length, sequence.query_dimension);
        detail::ValidateFiniteInput(
            sequence.k, sequence.kv_heads, sequence.key_length, sequence.query_dimension);
        detail::ValidateFiniteInput(
            sequence.v, sequence.kv_heads, sequence.key_length, sequence.value_dimension);
        std::vector<float> scores(std::min(key_chunk_size, sequence.key_length));
        std::vector<float> query(sequence.query_dimension);
        for(std::size_t head = 0; head < sequence.query_heads; ++head)
        {
            const auto kv_head = head / (sequence.query_heads / sequence.kv_heads);
            for(std::size_t selected_row = 0; selected_row < detail::SelectedRows(sequence);
                ++selected_row)
            {
                const auto row = detail::QueryRow(sequence, selected_row);
                ReferenceRow result{batch,
                                    head,
                                    row,
                                    0,
                                    std::vector<float>(sequence.value_dimension, 0),
                                    negative_infinity,
                                    std::vector<double>(sequence.value_dimension, 0),
                                    std::vector<double>(sequence.value_dimension, 0),
                                    0};
                for(std::size_t dim = 0; dim < sequence.query_dimension; ++dim)
                {
                    query[dim] = sequence.q(head, row, dim) * sequence.q_descale;
                    if(!std::isfinite(query[dim]))
                        throw std::runtime_error("FP32 attention query overflow");
                }
                float maximum = negative_infinity;
                float sum     = 0;
                for(std::size_t begin = 0; begin < sequence.key_length;)
                {
                    const auto count = std::min(key_chunk_size, sequence.key_length - begin);
                    float chunk_max  = negative_infinity;
                    for(std::size_t key = 0; key < count; ++key)
                    {
                        scores[key] = negative_infinity;
                        if(detail::IsMasked(sequence, row, begin + key))
                            continue;
                        float dot           = 0;
                        double absolute_dot = 0;
                        for(std::size_t dim = 0; dim < sequence.query_dimension; ++dim)
                        {
                            absolute_dot += std::abs(static_cast<double>(query[dim]) *
                                                     sequence.k(kv_head, begin + key, dim) *
                                                     sequence.k_descale);
                            dot =
                                std::fma(query[dim],
                                         sequence.k(kv_head, begin + key, dim) * sequence.k_descale,
                                         dot);
                        }
                        scores[key]           = dot * sequence.score_scale;
                        result.score_roundoff = std::max(
                            result.score_roundoff,
                            std::abs(static_cast<double>(sequence.score_scale)) *
                                    detail::Gamma(static_cast<double>(sequence.query_dimension)) *
                                    absolute_dot +
                                0x1p-24 * std::abs(static_cast<double>(scores[key])));
                        if(!std::isfinite(scores[key]))
                            throw std::runtime_error("FP32 attention score overflow");
                        chunk_max = std::max(chunk_max, scores[key]);
                        ++result.valid_keys;
                    }
                    if(chunk_max != negative_infinity)
                    {
                        const float next_max = std::max(maximum, chunk_max);
                        const float rescale  = sum == 0 ? 0 : std::exp(maximum - next_max);
                        sum *= rescale;
                        for(auto& value : result.output)
                            value *= rescale;
                        for(auto& value : result.absolute_contribution)
                            value *= rescale;
                        for(std::size_t key = 0; key < count; ++key)
                        {
                            if(scores[key] == negative_infinity)
                                continue;
                            const float probability = std::exp(scores[key] - next_max);
                            sum += probability;
                            for(std::size_t dim = 0; dim < sequence.value_dimension; ++dim)
                            {
                                const double absolute_value = std::abs(
                                    static_cast<double>(sequence.v(kv_head, begin + key, dim)) *
                                    sequence.v_descale);
                                result.absolute_contribution[dim] += probability * absolute_value;
                                result.maximum_value[dim] =
                                    std::max(result.maximum_value[dim], absolute_value);
                                result.output[dim] = std::fma(
                                    probability,
                                    sequence.v(kv_head, begin + key, dim) * sequence.v_descale,
                                    result.output[dim]);
                            }
                        }
                        maximum = next_max;
                    }
                    begin += count;
                }
                if(!sequence.sink_logits.empty() && std::isfinite(sequence.sink_logits[head]))
                {
                    const float sink     = sequence.sink_logits[head];
                    const float next_max = std::max(maximum, sink);
                    const float rescale  = sum == 0 ? 0 : std::exp(maximum - next_max);
                    sum                  = std::fma(sum, rescale, std::exp(sink - next_max));
                    for(auto& value : result.output)
                        value *= rescale;
                    for(auto& value : result.absolute_contribution)
                        value *= rescale;
                    maximum = next_max;
                }
                if(sum != 0)
                {
                    for(auto& value : result.output)
                        value /= sum;
                    for(auto& value : result.absolute_contribution)
                        value /= sum;
                    result.lse = maximum + std::log(sum);
                    if(!std::isfinite(result.lse))
                        throw std::runtime_error("FP32 attention LSE overflow");
                }
                for(const auto value : result.output)
                    if(!std::isfinite(value))
                        throw std::runtime_error("FP32 attention output overflow");
                reference.rows.push_back(std::move(result));
            }
        }
    }
    return reference;
}

// O finiteness is checked over every logical output, including unsampled rows.
// LSE permits -infinity only for a truly empty row; NaN/+infinity always fail.
// Cosine is 1 for two zero rows and 0 when exactly one row has zero norm.
inline Metrics Compare(const std::vector<Sequence>& sequences,
                       const Reference& reference,
                       const std::vector<Output>& outputs,
                       const Thresholds& thresholds                    = {},
                       const std::optional<AccuracyContract>& contract = std::nullopt)
{
    if(sequences.size() != outputs.size())
        throw std::invalid_argument("attention output sequence count mismatch");
    if(contract && ((contract->mantissa_bits != 7 && contract->mantissa_bits != 10) ||
                    (contract->key_tile_size != 64 && contract->key_tile_size != 128)))
        throw std::invalid_argument("unsupported attention accuracy contract");
    if(contract && reference.key_chunk_size == 0)
        throw std::invalid_argument("missing attention reference chunk size");
    if(!std::isfinite(thresholds.max_abs_error) || thresholds.max_abs_error < 0 ||
       !std::isfinite(thresholds.min_row_cosine) || thresholds.min_row_cosine < -1 ||
       thresholds.min_row_cosine > 1 || !std::isfinite(thresholds.mean_row_cosine) ||
       thresholds.mean_row_cosine < -1 || thresholds.mean_row_cosine > 1 ||
       !std::isfinite(thresholds.lse_atol) || thresholds.lse_atol < 0 ||
       !std::isfinite(thresholds.lse_rtol) || thresholds.lse_rtol < 0)
        throw std::invalid_argument("invalid attention comparison thresholds");
    Metrics metrics;
    double squared_error = 0;
    double squared_ref   = 0;
    double cosine_sum    = 0;
    std::size_t cursor   = 0;
    for(std::size_t batch = 0; batch < sequences.size(); ++batch)
    {
        const auto& sequence = sequences[batch];
        const auto& output   = outputs[batch];
        detail::ValidateSequence(sequence);
        detail::ValidateView(
            output.o, sequence.query_heads, sequence.query_length, sequence.value_dimension);
        if(output.lse)
            detail::ValidateView(*output.lse, sequence.query_heads, sequence.query_length, 1);
        for(std::size_t head = 0; head < sequence.query_heads; ++head)
        {
            double head_projection = 0;
            double head_ref_norm   = 0;
            for(std::size_t row = 0; row < sequence.query_length; ++row)
            {
                for(std::size_t dim = 0; dim < sequence.value_dimension; ++dim)
                {
                    ++metrics.scanned_output_elements;
                    metrics.nonfinite_output_count += !std::isfinite(output.o(head, row, dim));
                }
                if(output.lse)
                {
                    ++metrics.scanned_lse_rows;
                    const auto lse = (*output.lse)(head, row);
                    if(!std::isfinite(lse) && !(lse == -std::numeric_limits<float>::infinity() &&
                                                detail::IsEmptyRow(sequence, row, head)))
                        ++metrics.invalid_lse_count;
                }
            }
            for(std::size_t selected_row = 0; selected_row < detail::SelectedRows(sequence);
                ++selected_row)
            {
                const auto row = detail::QueryRow(sequence, selected_row);
                if(cursor >= reference.rows.size())
                    throw std::invalid_argument("attention reference rows are missing");
                const auto& expected = reference.rows[cursor++];
                if(expected.sequence != batch || expected.head != head || expected.query != row ||
                   expected.output.size() != sequence.value_dimension ||
                   expected.valid_keys > sequence.key_length)
                    throw std::invalid_argument("attention reference row identity mismatch");
                if(contract &&
                   (expected.absolute_contribution.size() != sequence.value_dimension ||
                    expected.maximum_value.size() != sequence.value_dimension ||
                    !std::isfinite(expected.score_roundoff) || expected.score_roundoff < 0))
                    throw std::invalid_argument("attention reference accuracy metadata is missing");
                ++metrics.checked_rows;
                double dot               = 0;
                double actual_norm       = 0;
                double ref_norm          = 0;
                bool finite_row          = true;
                bool component_mismatch  = false;
                double row_squared_error = 0;
                for(std::size_t dim = 0; dim < sequence.value_dimension; ++dim)
                {
                    ++metrics.checked_elements;
                    const double actual = output.o(head, row, dim);
                    const double target = expected.output[dim];
                    if(!std::isfinite(target))
                        throw std::invalid_argument("nonfinite attention reference output");
                    if(!std::isfinite(actual))
                    {
                        finite_row = false;
                        continue;
                    }
                    const double error = std::abs(actual - target);
                    if(contract)
                    {
                        if(!std::isfinite(expected.absolute_contribution[dim]) ||
                           expected.absolute_contribution[dim] < 0 ||
                           !std::isfinite(expected.maximum_value[dim]) ||
                           expected.maximum_value[dim] < 0)
                            throw std::invalid_argument("invalid attention reference contribution");
                        const double limit = detail::ComponentLimit(expected,
                                                                    dim,
                                                                    sequence.key_length,
                                                                    reference.key_chunk_size,
                                                                    *contract);
                        if(!std::isfinite(limit))
                            throw std::invalid_argument("nonfinite attention accuracy budget");
                        metrics.max_component_ratio = std::max(metrics.max_component_ratio,
                                                               detail::RelativeError(error, limit));
                        metrics.component_mismatch_count += error > limit;
                        component_mismatch |= error > limit;
                    }
                    metrics.max_abs_error      = std::max(metrics.max_abs_error, error);
                    metrics.max_relative_error = std::max(
                        metrics.max_relative_error, detail::RelativeError(error, std::abs(target)));
                    squared_error += error * error;
                    squared_ref += target * target;
                    dot += actual * target;
                    actual_norm += actual * actual;
                    ref_norm += target * target;
                    row_squared_error += error * error;
                    head_projection += (actual - target) * target;
                    head_ref_norm += target * target;
                }
                metrics.component_mismatch_rows += component_mismatch;
                metrics.max_row_relative_l2 =
                    std::max(metrics.max_row_relative_l2,
                             std::sqrt(detail::RelativeError(row_squared_error, ref_norm)));
                metrics.max_row_norm_drift = std::max(
                    metrics.max_row_norm_drift,
                    detail::RelativeError(std::abs(std::sqrt(actual_norm) - std::sqrt(ref_norm)),
                                          std::sqrt(ref_norm)));
                const double cosine =
                    !finite_row ? -1
                    : actual_norm == 0 || ref_norm == 0
                        ? (actual_norm == ref_norm ? 1 : 0)
                        : std::clamp(dot / std::sqrt(actual_norm * ref_norm), -1.0, 1.0);
                metrics.min_row_cosine = std::min(metrics.min_row_cosine, cosine);
                cosine_sum += cosine;
                if(output.lse)
                {
                    ++metrics.checked_lse_rows;
                    const double actual = (*output.lse)(head, row);
                    const double target = expected.lse;
                    const bool has_finite_sink =
                        !sequence.sink_logits.empty() && std::isfinite(sequence.sink_logits[head]);
                    if(expected.valid_keys == 0 && !has_finite_sink)
                    {
                        if(target != -std::numeric_limits<float>::infinity())
                            throw std::invalid_argument("empty reference row must have -inf LSE");
                        metrics.lse_mismatch_count += actual != target;
                    }
                    else
                    {
                        if(!std::isfinite(target))
                            throw std::invalid_argument("nonfinite nonempty reference LSE");
                        const double error        = std::isfinite(actual)
                                                        ? std::abs(actual - target)
                                                        : std::numeric_limits<double>::infinity();
                        metrics.max_lse_abs_error = std::max(metrics.max_lse_abs_error, error);
                        metrics.max_lse_relative_error =
                            std::max(metrics.max_lse_relative_error,
                                     detail::RelativeError(error, std::abs(target)));
                        metrics.lse_mismatch_count +=
                            error > thresholds.lse_atol + thresholds.lse_rtol * std::abs(target);
                    }
                }
            }
            metrics.max_head_gain =
                std::max(metrics.max_head_gain,
                         detail::RelativeError(std::abs(head_projection), head_ref_norm));
        }
    }
    if(cursor != reference.rows.size())
        throw std::invalid_argument("attention reference has unexpected rows");
    if(metrics.checked_elements != 0)
    {
        metrics.rms_error          = std::sqrt(squared_error / metrics.checked_elements);
        metrics.relative_rms_error = std::sqrt(detail::RelativeError(squared_error, squared_ref));
    }
    if(metrics.checked_rows != 0)
        metrics.mean_row_cosine = cosine_sum / metrics.checked_rows;
    if(metrics.nonfinite_output_count != 0)
    {
        metrics.max_abs_error      = std::numeric_limits<double>::infinity();
        metrics.max_relative_error = std::numeric_limits<double>::infinity();
        metrics.rms_error          = std::numeric_limits<double>::infinity();
        metrics.relative_rms_error = std::numeric_limits<double>::infinity();
    }
    metrics.strict_passed = metrics.nonfinite_output_count == 0 && metrics.invalid_lse_count == 0 &&
                            metrics.lse_mismatch_count == 0 &&
                            metrics.max_abs_error <= thresholds.max_abs_error &&
                            metrics.min_row_cosine >= thresholds.min_row_cosine &&
                            metrics.mean_row_cosine >= thresholds.mean_row_cosine;
    metrics.contract_passed = contract.has_value() && metrics.nonfinite_output_count == 0 &&
                              metrics.invalid_lse_count == 0 && metrics.lse_mismatch_count == 0 &&
                              metrics.max_abs_error <= thresholds.max_abs_error &&
                              metrics.component_mismatch_count == 0;
    metrics.passed =
        contract ? metrics.contract_passed && (!contract->RequiresStrict() || metrics.strict_passed)
                 : metrics.strict_passed;
    return metrics;
}

} // namespace reference
} // namespace ck_tile::test
