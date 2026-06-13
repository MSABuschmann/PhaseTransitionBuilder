// Simple linear interpolation by binary search.
// Carried from the reference BubbleMaster implementation.

#pragma once

#include <algorithm>
#include <vector>

class Interpolator {
public:
    Interpolator(const std::vector<double> &x, const std::vector<double> &y)
        : x_(x), y_(y) {}

    double operator()(double xi) const {
        if (xi <= x_.front()) return y_.front();
        if (xi >= x_.back())  return y_.back();
        auto it  = std::lower_bound(x_.begin(), x_.end(), xi);
        int  idx = std::max(int(it - x_.begin()) - 1, 0);
        double x0 = x_[idx], x1 = x_[idx + 1];
        double y0 = y_[idx], y1 = y_[idx + 1];
        return y0 + (xi - x0) * (y1 - y0) / (x1 - x0);
    }

private:
    const std::vector<double> &x_;
    const std::vector<double> &y_;
};
