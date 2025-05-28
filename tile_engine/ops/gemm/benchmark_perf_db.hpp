// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <sqlite3.h>
#include <tuple>
#include <memory>

#include "benchmark_gemm.hpp"

#define CHECK_SQLITE3(expr, db)                                                               \
    do                                                                                        \
    {                                                                                         \
        int result_code = (expr);                                                             \
        if(result_code != SQLITE_OK)                                                          \
        {                                                                                     \
            const char* err = sqlite3_errmsg(db);                                             \
            throw std::runtime_error("SQLite error[" + std::to_string(result_code) +          \
                                     "]: " + (err ? err : "unknown error") + " at " +         \
                                     std::string(__FILE__) + ":" + std::to_string(__LINE__)); \
        }                                                                                     \
    } while(0)

#define CHECK_SQLITE3_RC(expr, db, rc)                                                        \
    do                                                                                        \
    {                                                                                         \
        rc = (expr);                                                                          \
        if(rc != SQLITE_OK && rc != SQLITE_ROW && rc != SQLITE_DONE)                          \
        {                                                                                     \
            const char* err = sqlite3_errmsg(db);                                             \
            throw std::runtime_error("SQLite error[" + std::to_string(rc) +                   \
                                     "]: " + (err ? err : "unknown error") + " at " +         \
                                     std::string(__FILE__) + ":" + std::to_string(__LINE__)); \
        }                                                                                     \
    } while(0)

class StmtWrapper
{
    public:
    explicit StmtWrapper(sqlite3* db, const char* sql)
        : stmt_(
              [db, sql] {
                  sqlite3_stmt* stmt = nullptr;
                  CHECK_SQLITE3(sqlite3_prepare_v2(db, sql, -1, &stmt, nullptr), db);
                  return stmt;
              }(),
              &sqlite3_finalize)
    {
    }

    operator sqlite3_stmt*() const { return stmt_.get(); }

    private:
    std::unique_ptr<sqlite3_stmt, decltype(&sqlite3_finalize)> stmt_;
};

class BenchmarkPerfDB
{
    public:
    explicit BenchmarkPerfDB(const std::filesystem::path& path)
        : db_ptr_(
              [path] {
                  sqlite3* raw_db_ptr = nullptr;
                  CHECK_SQLITE3(sqlite3_open_v2(path.string().c_str(),
                                                &raw_db_ptr,
                                                SQLITE_OPEN_READWRITE | SQLITE_OPEN_CREATE,
                                                nullptr),
                                raw_db_ptr);
                  return raw_db_ptr;
              }(),
              &sqlite3_close)
    {

        try
        {
            execute("PRAGMA journal_mode = WAL");
            execute("PRAGMA synchronous = NORMAL");
            execute("PRAGMA foreign_keys = ON");

            constexpr const char* schema = R"sql(
                CREATE TABLE IF NOT EXISTS gemm (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rocm_version TEXT NOT NULL CHECK(length(rocm_version) > 0),
                    device_name TEXT NOT NULL CHECK(length(device_name) > 0),
                    problem TEXT NOT NULL CHECK(json_valid(problem)),
                    instance_name TEXT NOT NULL CHECK(length(instance_name) > 0),
                    latency REAL CHECK(latency > 0),
                    tflops REAL CHECK(tflops > 0),
                    bandwidth REAL CHECK(bandwidth > 0)
                );
            )sql";

            execute(schema);
        }
        catch(...)
        {
            throw;
        }
    }

    bool check_if_record_problem(const std::string& rocm_version,
                                 const std::string& device_name,
                                 const GemmProblem& gemm_problem)
    {
        constexpr const char* sql = R"sql(
            SELECT 1 FROM gemm
            WHERE rocm_version=? 
            AND device_name=?
            AND problem=?
            LIMIT 1
        )sql";

        StmtWrapper stmt(db_ptr_.get(), sql);
        sqlite3_stmt* raw_stmt = stmt;
        int idx                = 1;
        CHECK_SQLITE3(
            sqlite3_bind_text(raw_stmt, idx++, rocm_version.c_str(), -1, SQLITE_TRANSIENT),
            db_ptr_.get());
        CHECK_SQLITE3(sqlite3_bind_text(raw_stmt, idx++, device_name.c_str(), -1, SQLITE_TRANSIENT),
                      db_ptr_.get());
        CHECK_SQLITE3(sqlite3_bind_text(
                          raw_stmt, idx++, gemm_problem.to_json().c_str(), -1, SQLITE_TRANSIENT),
                      db_ptr_.get());

        int rc;
        CHECK_SQLITE3_RC(sqlite3_step(raw_stmt), db_ptr_.get(), rc);
        CHECK_SQLITE3(sqlite3_reset(raw_stmt), db_ptr_.get());
        CHECK_SQLITE3(sqlite3_clear_bindings(raw_stmt), db_ptr_.get());

        if(rc == SQLITE_DONE)
        {
            std::cout << "No matching records found" << std::endl;
        }
        return (rc == SQLITE_ROW);
    }

    std::tuple<std::string, PerformanceResult> query_cache(const std::string& rocm_version,
                                                           const std::string& device_name,
                                                           const GemmProblem& gemm_problem)
    {
        constexpr const char* sql = R"sql(
            SELECT instance_name, latency, tflops, bandwidth FROM gemm 
            WHERE rocm_version=? 
            AND device_name=?
            AND problem=?
            LIMIT 1
        )sql";

        StmtWrapper stmt(db_ptr_.get(), sql);
        sqlite3_stmt* raw_stmt = stmt;

        int idx = 1;
        CHECK_SQLITE3(
            sqlite3_bind_text(raw_stmt, idx++, rocm_version.c_str(), -1, SQLITE_TRANSIENT),
            db_ptr_.get());
        CHECK_SQLITE3(sqlite3_bind_text(raw_stmt, idx++, device_name.c_str(), -1, SQLITE_TRANSIENT),
                      db_ptr_.get());
        CHECK_SQLITE3(
            sqlite3_bind_text(stmt, idx++, gemm_problem.to_json().c_str(), -1, SQLITE_TRANSIENT),
            db_ptr_.get());

        int rc;
        CHECK_SQLITE3_RC(sqlite3_step(raw_stmt), db_ptr_.get(), rc);

        if(rc == SQLITE_ROW)
        {
            return std::make_tuple(reinterpret_cast<const char*>(sqlite3_column_text(raw_stmt, 0)),
                                   PerformanceResult{sqlite3_column_double(raw_stmt, 1),
                                                     sqlite3_column_double(raw_stmt, 2),
                                                     sqlite3_column_double(raw_stmt, 3)});
        }
        else if(rc != SQLITE_DONE)
        {
            throw std::runtime_error(sqlite3_errmsg(db_ptr_.get()));
        }
        CHECK_SQLITE3(sqlite3_reset(raw_stmt), db_ptr_.get());
        CHECK_SQLITE3(sqlite3_clear_bindings(raw_stmt), db_ptr_.get());

        return std::make_tuple("", PerformanceResult{-1.0f, -1.0f, -1.0f});
    }

    void insert_cache(const std::string& rocm_version,
                      const std::string& device_name,
                      const std::vector<KernelInstance>& kernen_instnaces)
    {
        execute("BEGIN TRANSACTION");
        try
        {
            constexpr const char* sql = R"sql(
                INSERT INTO gemm
                    (rocm_version, 
                     device_name, 
                     problem, 
                     instance_name, 
                     latency, 
                     tflops, 
                     bandwidth)
                VALUES (?1, 
                        ?2, 
                        ?3, 
                        ?4, 
                        ?5, 
                        ?6, 
                        ?7)
            )sql";

            StmtWrapper stmt(db_ptr_.get(), sql);
            sqlite3_stmt* raw_stmt = stmt;

            for(const auto& item : kernen_instnaces)
            {
                int idx = 1;
                CHECK_SQLITE3(
                    sqlite3_bind_text(raw_stmt, idx++, rocm_version.c_str(), -1, SQLITE_TRANSIENT),
                    db_ptr_.get());
                CHECK_SQLITE3(
                    sqlite3_bind_text(raw_stmt, idx++, device_name.c_str(), -1, SQLITE_TRANSIENT),
                    db_ptr_.get());
                CHECK_SQLITE3(
                    sqlite3_bind_text(
                        raw_stmt, idx++, item.problem_.to_json().c_str(), -1, SQLITE_TRANSIENT),
                    db_ptr_.get());
                CHECK_SQLITE3(
                    sqlite3_bind_text(raw_stmt, idx++, item.name_.c_str(), -1, SQLITE_TRANSIENT),
                    db_ptr_.get());
                CHECK_SQLITE3(sqlite3_bind_double(raw_stmt, idx++, item.perf_result_.latency_),
                              db_ptr_.get());
                CHECK_SQLITE3(sqlite3_bind_double(raw_stmt, idx++, item.perf_result_.tflops_),
                              db_ptr_.get());
                CHECK_SQLITE3(sqlite3_bind_double(raw_stmt, idx++, item.perf_result_.bandwidth_),
                              db_ptr_.get());

                int rc;
                CHECK_SQLITE3_RC(sqlite3_step(raw_stmt), db_ptr_.get(), rc);
                CHECK_SQLITE3(sqlite3_reset(raw_stmt), db_ptr_.get());
                CHECK_SQLITE3(sqlite3_clear_bindings(raw_stmt), db_ptr_.get());
            }
            execute("COMMIT");
        }
        catch(...)
        {
            execute("ROLLBACK");
            throw;
        }
    }

    private:
    void execute(const char* sql)
    {
        CHECK_SQLITE3(sqlite3_exec(db_ptr_.get(), sql, nullptr, nullptr, nullptr), db_ptr_.get());
    }

    std::unique_ptr<sqlite3, decltype(&sqlite3_close)> db_ptr_;
};
