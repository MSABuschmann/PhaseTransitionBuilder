#pragma once

#include <functional>
#include <utility>
#include <vector>

// Turns a sequential stream of Evolution snapshots into batches of up to
// `batch_size` NEW slices, each batch (after the first) prefixed with a
// one-slice halo carried from the previous batch's last new slice — needed
// by precompute_z_kernel's i_s-1 finite-difference derivative in s.
//
// Handles: batch 0 (no halo); a short final batch; and the degenerate case
// where the very first Push() is already the last snapshot (tiny gamma_ij),
// which correctly reports is_first_batch == is_last_batch == true.
class SBatchWindow {
public:
    using BatchReady =
        std::function<void(const std::vector<std::vector<double>> &phi_batch,
                            const std::vector<double> &s_batch,
                            bool is_first_batch, bool is_last_batch)>;

    SBatchWindow(int batch_size, BatchReady cb)
        : batch_size_(batch_size), cb_(std::move(cb)) {}

    // Call once per Evolution snapshot, in increasing-s order.
    void Push(const std::vector<double> &phi, double s, bool is_last_snapshot) {
        phi_buf_.push_back(phi);
        s_buf_.push_back(s);

        const int new_count =
            static_cast<int>(phi_buf_.size()) - (first_batch_done_ ? 1 : 0);
        if (new_count < batch_size_ && !is_last_snapshot)
            return;

        cb_(phi_buf_, s_buf_, !first_batch_done_, is_last_snapshot);
        first_batch_done_ = true;

        if (is_last_snapshot) {
            phi_buf_.clear();
            s_buf_.clear();
            return;
        }

        // Carry the last slice forward as the next batch's halo.
        std::vector<double> halo_phi = std::move(phi_buf_.back());
        double halo_s = s_buf_.back();
        phi_buf_.clear();
        s_buf_.clear();
        phi_buf_.push_back(std::move(halo_phi));
        s_buf_.push_back(halo_s);
    }

private:
    int batch_size_;
    BatchReady cb_;
    bool first_batch_done_ = false;
    std::vector<std::vector<double>> phi_buf_;
    std::vector<double> s_buf_;
};
