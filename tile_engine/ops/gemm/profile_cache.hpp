// SPDX-License-Identifier: MIT
// Copyright (c) 2025, Advanced Micro Devices, Inc. All rights reserved.

#include <sqlite3.h>

#include "gemm_host_api.hpp"

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

class ProfileCacheDB
{
    public:
    explicit ProfileCacheDB(const std::filesystem::path& path)
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
            exec_direct("PRAGMA journal_mode = WAL");
            exec_direct("PRAGMA synchronous = NORMAL");
            exec_direct("PRAGMA foreign_keys = ON");

            constexpr const char* schema = R"sql(
                CREATE TABLE IF NOT EXISTS gemm (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rocm_version TEXT NOT NULL CHECK(length(rocm_version) > 0),
                    commit_id TEXT NOT NULL CHECK(length(commit_id) > 0),
                    device_name TEXT NOT NULL CHECK(length(device_name) > 0),
                    instance_name TEXT NOT NULL CHECK(length(instance_name) > 0),
                    problem TEXT NOT NULL CHECK(json_valid(problem)),
                    latency REAL CHECK(latency > 0),
                    tflops REAL CHECK(tflops > 0),
                    bandwidth REAL CHECK(bandwidth > 0)
                );
                CREATE INDEX IF NOT EXISTS idx_latency ON gemm(latency);
                CREATE INDEX IF NOT EXISTS idx_tflops_desc ON gemm(tflops DESC);
                CREATE INDEX IF NOT EXISTS idx_bandwidth_desc ON gemm(bandwidth DESC);
            )sql";

            exec_direct(schema);
        }
        catch(...)
        {
            throw;
        }
    }

    bool check_if_record(const KernelInstance& kernel_instance)
    {
        constexpr const char* sql = R"sql(
    SELECT 1 FROM gemm 
    WHERE rocm_version=? AND commit_id=? AND device_name=?
      AND instance_name=? AND problem=?
    LIMIT 1)sql";

        StmtWrapper stmt(db_ptr_.get(), sql);
        sqlite3_stmt* raw_stmt = stmt;
        int idx                = 1;
        CHECK_SQLITE3(
            sqlite3_bind_text(
                raw_stmt, idx++, kernel_instance.env.rocm_version.c_str(), -1, SQLITE_TRANSIENT),
            db_ptr_.get());
        CHECK_SQLITE3(
            sqlite3_bind_text(
                raw_stmt, idx++, kernel_instance.env.commit_id.c_str(), -1, SQLITE_TRANSIENT),
            db_ptr_.get());
        CHECK_SQLITE3(
            sqlite3_bind_text(
                raw_stmt, idx++, kernel_instance.env.device_name.c_str(), -1, SQLITE_TRANSIENT),
            db_ptr_.get());
        CHECK_SQLITE3(
            sqlite3_bind_text(raw_stmt, idx++, kernel_instance.name.c_str(), -1, SQLITE_TRANSIENT),
            db_ptr_.get());
        CHECK_SQLITE3(sqlite3_bind_text(raw_stmt,
                                        idx++,
                                        kernel_instance.problem.serialize().c_str(),
                                        kernel_instance.problem.serialize().size(),
                                        SQLITE_TRANSIENT),
                      db_ptr_.get());

        int rc;
        CHECK_SQLITE3_RC(sqlite3_step(raw_stmt), db_ptr_.get(), rc);
        CHECK_SQLITE3(sqlite3_reset(raw_stmt), db_ptr_.get());
        CHECK_SQLITE3(sqlite3_clear_bindings(raw_stmt), db_ptr_.get());
        return (rc == SQLITE_ROW);
    }

    PerformanceResult query_performance_result(const KernelInstance& kernel_instance)
    {
        constexpr const char* sql = R"sql(
        SELECT latency, tflops, bandwidth FROM gemm 
        WHERE rocm_version=? AND commit_id=? AND device_name=?
        AND instance_name=? AND problem=?
        LIMIT 1)sql";

        StmtWrapper stmt(db_ptr_.get(), sql);
        sqlite3_stmt* raw_stmt = stmt;

        int idx = 1;
        CHECK_SQLITE3(
            sqlite3_bind_text(
                raw_stmt, idx++, kernel_instance.env.rocm_version.c_str(), -1, SQLITE_TRANSIENT),
            db_ptr_.get());
        CHECK_SQLITE3(
            sqlite3_bind_text(
                raw_stmt, idx++, kernel_instance.env.commit_id.c_str(), -1, SQLITE_TRANSIENT),
            db_ptr_.get());
        CHECK_SQLITE3(
            sqlite3_bind_text(
                raw_stmt, idx++, kernel_instance.env.device_name.c_str(), -1, SQLITE_TRANSIENT),
            db_ptr_.get());
        CHECK_SQLITE3(
            sqlite3_bind_text(raw_stmt, idx++, kernel_instance.name.c_str(), -1, SQLITE_TRANSIENT),
            db_ptr_.get());
        CHECK_SQLITE3(sqlite3_bind_text(stmt,
                                        idx++,
                                        kernel_instance.problem.serialize().c_str(),
                                        kernel_instance.problem.serialize().size(),
                                        SQLITE_TRANSIENT),
                      db_ptr_.get());

        int rc;
        CHECK_SQLITE3_RC(sqlite3_step(raw_stmt), db_ptr_.get(), rc);

        if(rc == SQLITE_ROW)
        {
            return {sqlite3_column_double(raw_stmt, 0),
                    sqlite3_column_double(raw_stmt, 1),
                    sqlite3_column_double(raw_stmt, 2)};
        }
        else if(rc != SQLITE_DONE)
        {
            throw std::runtime_error(sqlite3_errmsg(db_ptr_.get()));
        }
        CHECK_SQLITE3(sqlite3_reset(raw_stmt), db_ptr_.get());
        CHECK_SQLITE3(sqlite3_clear_bindings(raw_stmt), db_ptr_.get());

        return {-1.0f, -1.0f, -1.0f};
    }

    void insert_batch(const std::vector<KernelInstance>& data)
    {
        exec_direct("BEGIN TRANSACTION");
        try
        {
            constexpr const char* sql = R"sql(
            INSERT INTO gemm
                (rocm_version, commit_id, device_name, 
                 instance_name, problem, 
                 latency, tflops, bandwidth)
            VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8))sql";

            StmtWrapper stmt(db_ptr_.get(), sql);
            sqlite3_stmt* raw_stmt = stmt;

            for(const auto& item : data)
            {
                int idx = 1;
                CHECK_SQLITE3(
                    sqlite3_bind_text(
                        raw_stmt, idx++, item.env.rocm_version.c_str(), -1, SQLITE_TRANSIENT),
                    db_ptr_.get());
                CHECK_SQLITE3(
                    sqlite3_bind_text(
                        raw_stmt, idx++, item.env.commit_id.c_str(), -1, SQLITE_TRANSIENT),
                    db_ptr_.get());
                CHECK_SQLITE3(
                    sqlite3_bind_text(
                        raw_stmt, idx++, item.env.device_name.c_str(), -1, SQLITE_TRANSIENT),
                    db_ptr_.get());
                CHECK_SQLITE3(
                    sqlite3_bind_text(raw_stmt, idx++, item.name.c_str(), -1, SQLITE_TRANSIENT),
                    db_ptr_.get());
                CHECK_SQLITE3(sqlite3_bind_text(raw_stmt,
                                                idx++,
                                                item.problem.serialize().c_str(),
                                                item.problem.serialize().size(),
                                                SQLITE_TRANSIENT),
                              db_ptr_.get());
                CHECK_SQLITE3(sqlite3_bind_double(raw_stmt, idx++, item.perf_result.latency),
                              db_ptr_.get());
                CHECK_SQLITE3(sqlite3_bind_double(raw_stmt, idx++, item.perf_result.tflops),
                              db_ptr_.get());
                CHECK_SQLITE3(sqlite3_bind_double(raw_stmt, idx++, item.perf_result.bandwidth),
                              db_ptr_.get());

                int rc;
                CHECK_SQLITE3_RC(sqlite3_step(raw_stmt), db_ptr_.get(), rc);
                CHECK_SQLITE3(sqlite3_reset(raw_stmt), db_ptr_.get());
                CHECK_SQLITE3(sqlite3_clear_bindings(raw_stmt), db_ptr_.get());
            }
            exec_direct("COMMIT");
        }
        catch(...)
        {
            exec_direct("ROLLBACK");
            throw;
        }
    }

    // std::vector<KernelInstance> query_top(Metric metric, int limit = 5)
    // {
    //     if(limit <= 0)
    //         throw invalid_argument("Limit must be positive");

    //     const char* order = nullptr;
    //     switch(metric)
    //     {
    //     case LATENCY: order = "latency ASC"; break;
    //     case TFLOPS: order = "tflops DESC"; break;
    //     case BANDWIDTH: order = "bandwidth DESC"; break;
    //     default: throw invalid_argument("Invalid metric");
    //     }

    //     string sql = "SELECT name, latency, tflops, bandwidth FROM kernels "
    //                  "ORDER BY " +
    //                  string(order) + " LIMIT ?";

    //     StmtWrapper stmt(db, sql.c_str());
    //     sqlite3_bind_int(stmt, 1, limit);

    //     std::vector<KernelInstance> results;
    //     while(sqlite3_step(stmt) == SQLITE_ROW)
    //     {
    //         results.emplace_back(reinterpret_cast<const char*>(sqlite3_column_text(stmt, 0)),
    //                              PerformanceResult{sqlite3_column_double(stmt, 1),
    //                                                sqlite3_column_double(stmt, 2),
    //                                                sqlite3_column_double(stmt, 3)});
    //     }
    //     return results;
    // }

    private:
    void exec_direct(const char* sql)
    {
        CHECK_SQLITE3(sqlite3_exec(db_ptr_.get(), sql, nullptr, nullptr, nullptr), db_ptr_.get());
    }

    std::unique_ptr<sqlite3, decltype(&sqlite3_close)> db_ptr_;
};
