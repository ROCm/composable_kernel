// Copyright (c) Advanced Micro Devices, Inc., or its affiliates.
// SPDX-License-Identifier: MIT

#include "ck/host/ck_tile_headers_preprocessor.hpp"

#include <algorithm>
#include <cctype>
#include <functional>
#include <string_view>
#include <vector>

// Adapted from migraphx

namespace ck {
namespace host {

static constexpr std::string_view HOST_TOKEN                 = "CK_TILE_HOST";
static constexpr std::string_view REPLACEMENT                = "{ __builtin_unreachable(); }";
static constexpr std::string_view CONSTEXPR_REPLACEMENT      = "{ return {}; }";
static constexpr std::string_view CONSTEXPR_AUTO_REPLACEMENT = "{ return 0; }";

enum class TokenType
{
    StringLiteral,
    CharLiteral,
    Number,
    Comment,
    Whitespace,
    Identifier,
    Punctuation
};

struct token
{
    std::string_view text;
    TokenType type;
};

using lexer_fn = std::function<const char*(const char* start, const char* end)>;

template <class P>
static lexer_fn lex_while(P p)
{
    return [=](const char* start, const char* end) {
        return std::find_if(start, end, [&](char c) { return !p(c); });
    };
}

struct tagged_lexer
{
    lexer_fn fn;
    TokenType type;
};

static std::vector<token>
tokenize(const char* start, const char* end, const std::vector<tagged_lexer>& lexers)
{
    std::vector<token> tokens;
    while(start != end)
    {
        bool matched = false;
        for(const auto& lex : lexers)
        {
            const char* next = lex.fn(start, end);
            if(next != start)
            {
                tokens.push_back({std::string_view(start, next - start), lex.type});
                start   = next;
                matched = true;
                break;
            }
        }
        if(!matched)
        {
            tokens.push_back({std::string_view(start, 1), TokenType::Punctuation});
            ++start;
        }
    }
    return tokens;
}

static std::vector<token> cpp_tokenize(std::string_view s)
{
    std::vector<tagged_lexer> lexers;

    // Raw string literal: R"delim(...)delim"
    lexers.push_back({[](const char* start, const char* end) -> const char* {
                          if(*start != 'R' || start + 1 >= end || start[1] != '"')
                              return start;
                          const char* p           = start + 2;
                          const char* delim_start = p;
                          while(p < end && *p != '(')
                              ++p;
                          if(p >= end)
                              return start;
                          size_t delim_len = static_cast<size_t>(p - delim_start);
                          ++p;
                          while(p < end)
                          {
                              if(*p == ')' && p + delim_len + 1 < end &&
                                 std::equal(delim_start, delim_start + delim_len, p + 1) &&
                                 p[delim_len + 1] == '"')
                              {
                                  return p + delim_len + 2;
                              }
                              ++p;
                          }
                          return start;
                      },
                      TokenType::StringLiteral});

    // String literal: "..."
    lexers.push_back({[](const char* start, const char* end) -> const char* {
                          if(*start != '"')
                              return start;
                          const char* p = start + 1;
                          while(p < end && *p != '"')
                          {
                              if(*p == '\\')
                                  ++p;
                              ++p;
                          }
                          return (p < end) ? p + 1 : start;
                      },
                      TokenType::StringLiteral});

    // Numeric literal (must precede char literal to handle digit separators like 0b1100'1111)
    lexers.push_back({[](const char* start, const char* end) -> const char* {
                          if(!std::isdigit(static_cast<unsigned char>(*start)))
                              return start;
                          const char* p = start + 1;
                          while(p < end && (std::isalnum(static_cast<unsigned char>(*p)) ||
                                            *p == '\'' || *p == '.'))
                              ++p;
                          return p;
                      },
                      TokenType::Number});

    // Char literal: '...'
    lexers.push_back({[](const char* start, const char* end) -> const char* {
                          if(*start != '\'')
                              return start;
                          const char* p = start + 1;
                          while(p < end && *p != '\'')
                          {
                              if(*p == '\\')
                                  ++p;
                              ++p;
                          }
                          return (p < end) ? p + 1 : start;
                      },
                      TokenType::CharLiteral});

    // Block comment: /* ... */  (must come before line comment)
    lexers.push_back({[](const char* start, const char* end) -> const char* {
                          if(start + 1 >= end || start[0] != '/' || start[1] != '*')
                              return start;
                          const char* p = start + 2;
                          while(p + 1 < end)
                          {
                              if(p[0] == '*' && p[1] == '/')
                                  return p + 2;
                              ++p;
                          }
                          return start;
                      },
                      TokenType::Comment});

    // Line comment: // ...
    lexers.push_back({[](const char* start, const char* end) -> const char* {
                          if(start + 1 >= end || start[0] != '/' || start[1] != '/')
                              return start;
                          return std::find(start + 2, end, '\n');
                      },
                      TokenType::Comment});

    // Whitespace
    lexers.push_back({lex_while([](char c) { return std::isspace(static_cast<unsigned char>(c)); }),
                      TokenType::Whitespace});

    // Identifier / keyword
    lexers.push_back(
        {lex_while([](char c) { return std::isalnum(static_cast<unsigned char>(c)) || c == '_'; }),
         TokenType::Identifier});

    // Single punctuation character (catch-all)
    lexers.push_back({[](const char* start, const char*) -> const char* { return start + 1; },
                      TokenType::Punctuation});

    return tokenize(s.data(), s.data() + s.size(), lexers);
}

// Check if the token at `idx` sits on a #define line by scanning backwards.
static bool is_on_define_line(const std::vector<token>& tokens, size_t idx)
{
    for(size_t j = idx; j-- > 0;)
    {
        const auto& t = tokens[j];

        if(t.type == TokenType::Whitespace && t.text.find('\n') != std::string_view::npos)
            return false;

        if(t.type == TokenType::Whitespace)
            continue;

        if(t.text != "define")
            return false;

        for(size_t k = j; k-- > 0;)
        {
            const auto& u = tokens[k];
            if(u.type == TokenType::Whitespace && u.text.find('\n') != std::string_view::npos)
                return false;
            if(u.type == TokenType::Whitespace)
                continue;
            return u.text == "#";
        }
        return false;
    }
    return false;
}

// Starting from token index `open_idx` (which must be a "{" token), find the
// index of the matching "}".
static size_t find_matching_brace(const std::vector<token>& tokens, size_t open_idx)
{
    int depth = 1;
    for(size_t i = open_idx + 1; i < tokens.size() && depth > 0; ++i)
    {
        if(tokens[i].text == "{")
            ++depth;
        else if(tokens[i].text == "}")
            --depth;

        if(depth == 0)
            return i;
    }
    return std::string::npos;
}

static std::string_view choose_replacement(bool is_constexpr, bool is_auto)
{
    if(is_constexpr && is_auto)
        return CONSTEXPR_AUTO_REPLACEMENT;
    if(is_constexpr)
        return CONSTEXPR_REPLACEMENT;
    return REPLACEMENT;
}

std::string strip_host_bodies(std::string_view content)
{
    auto tokens = cpp_tokenize(content);

    std::string result;
    result.reserve(content.size());

    for(size_t i = 0; i < tokens.size(); ++i)
    {
        if(tokens[i].text != HOST_TOKEN || is_on_define_line(tokens, i))
        {
            result.append(tokens[i].text);
            continue;
        }

        result.append(tokens[i].text);

        // Scan forward past the signature to find '{' or ';'
        bool is_constexpr   = false;
        bool is_auto_return = false;
        int paren_depth     = 0;
        size_t j            = i + 1;

        for(; j < tokens.size(); ++j)
        {
            const auto& t = tokens[j];

            if(t.type == TokenType::Whitespace || t.type == TokenType::Comment)
                continue;

            if(paren_depth == 0 && t.text == "constexpr")
                is_constexpr = true;
            if(paren_depth == 0 && t.text == "auto")
                is_auto_return = true;

            if(t.text == "(")
                ++paren_depth;
            else if(t.text == ")")
                --paren_depth;
            else if(paren_depth == 0 && t.text == ";")
                break;
            else if(paren_depth == 0 && t.text == "{")
                break;
        }

        if(j >= tokens.size() || tokens[j].text == ";")
        {
            for(size_t k = i + 1; k <= j && k < tokens.size(); ++k)
                result.append(tokens[k].text);
            i = j;
            continue;
        }

        size_t close = find_matching_brace(tokens, j);

        for(size_t k = i + 1; k < j; ++k)
            result.append(tokens[k].text);

        if(close == std::string::npos)
        {
            for(size_t k = j; k < tokens.size(); ++k)
                result.append(tokens[k].text);
            i = tokens.size();
            break;
        }

        result.append(choose_replacement(is_constexpr, is_auto_return));
        i = close;
    }

    return result;
}

} // namespace host
} // namespace ck
