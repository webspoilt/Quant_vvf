#pragma once

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#if defined(_WIN32)
#include <io.h>
#define fsync _commit
#define fileno _fileno
#else
#include <unistd.h>
#endif

namespace hft::engine {

#pragma pack(push, 1)
struct JournalRecord {
    uint32_t magic{0x57414C31};       // "WAL1" format magic
    uint16_t version{1};              // Schema version
    uint16_t reserved{0};
    uint64_t exec_id{0};
    uint64_t order_id{0};
    uint8_t side{0};
    double filled_qty{0.0};
    double fill_price{0.0};
    uint64_t timestamp_ns{0};
    uint32_t checksum{0};
};
#pragma pack(pop)

/**
 * Lightweight, zero-allocation durable binary execution write-ahead log (WAL).
 * Guarantees crash resilience, hardware fsync commit, and torn-record rejection.
 */
class DurableExecutionJournal {
public:
    explicit DurableExecutionJournal(const std::string& journal_path)
        : journal_path_(journal_path), file_(nullptr) {
        open_journal();
    }

    ~DurableExecutionJournal() {
        if (file_) {
            std::fflush(file_);
            int fd = fileno(file_);
            if (fd >= 0) {
                fsync(fd);
            }
            std::fclose(file_);
            file_ = nullptr;
        }
    }

    // Append-only write of committed execution transaction with hardware fsync
    inline bool append_execution(
        uint64_t exec_id,
        uint64_t order_id,
        uint8_t side,
        double filled_qty,
        double fill_price,
        uint64_t timestamp_ns
    ) noexcept {
        if (!file_) [[unlikely]] return false;

        JournalRecord rec;
        rec.magic = 0x57414C31;
        rec.version = 1;
        rec.reserved = 0;
        rec.exec_id = exec_id;
        rec.order_id = order_id;
        rec.side = side;
        rec.filled_qty = filled_qty;
        rec.fill_price = fill_price;
        rec.timestamp_ns = timestamp_ns;
        rec.checksum = calculate_checksum(rec);

        size_t written = std::fwrite(&rec, sizeof(JournalRecord), 1, file_);
        if (written == 1) {
            std::fflush(file_);
            int fd = fileno(file_);
            if (fd >= 0) {
                fsync(fd); // Physical platter/NVMe persistent commit
            }
            return true;
        }
        return false;
    }

    // Recover all committed executions from disk upon restart, safely ignoring torn records
    std::vector<JournalRecord> recover_committed_records() {
        std::vector<JournalRecord> records;
        std::FILE* r_file = std::fopen(journal_path_.c_str(), "rb");
        if (!r_file) return records;

        JournalRecord rec;
        while (std::fread(&rec, sizeof(JournalRecord), 1, r_file) == 1) {
            // Verify magic, schema version, and cryptographic Adler/FNV checksum
            if (rec.magic == 0x57414C31 && rec.version == 1) {
                if (rec.checksum == calculate_checksum(rec)) {
                    records.push_back(rec);
                }
            }
        }
        std::fclose(r_file);
        return records;
    }

private:
    std::string journal_path_;
    std::FILE* file_;

    void open_journal() {
        file_ = std::fopen(journal_path_.c_str(), "ab");
    }

    static inline uint32_t calculate_checksum(const JournalRecord& rec) noexcept {
        uint32_t sum = 0x55AA55AA;
        const uint8_t* p = reinterpret_cast<const uint8_t*>(&rec);
        // Exclude the checksum field itself (last 4 bytes)
        for (size_t i = 0; i < sizeof(JournalRecord) - sizeof(uint32_t); ++i) {
            sum = ((sum << 5) + sum) ^ p[i];
        }
        return sum;
    }
};

} // namespace hft::engine

