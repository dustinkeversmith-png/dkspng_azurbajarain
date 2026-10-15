#pragma once

#include <cstddef>

namespace forward_analysis {

struct OutputObservation {
    // Output index k.
    std::size_t index = 0;
    // True when the prediction has the same sign as the target on every sample.
    bool sign_preserved = false;
    // True when the predictions order the samples exactly as the targets do.
    bool order_preserved = false;
    // sum |prediction| / sum |target| (below 1 means damped, above 1 means amplified).
    double amplitude_ratio = 1.0;
    // Pearson correlation between prediction and target across samples.
    double correlation = 0.0;
    // True when the prediction's spread is at most flat_output_ratio of the target's spread.
    bool flat = false;
    // True when the prediction is one identical value on every gated (closed) sample.
    bool constant_on_closed_samples = false;
    // Prediction/target correlation over the open (non-gated) samples only; equals correlation when nothing is gated.
    double open_side_correlation = 0.0;
    // Share of the absolute error that falls on closed samples (0 when nothing is gated).
    double closed_error_share = 0.0;
    // (max - min) of the prediction over (max - min) of the target; 1 when the target has no range.
    double range_ratio = 1.0;
    // Samples on which sign(prediction) equals sign(target), with 0 its own sign.
    std::size_t sign_agreements = 0;
};

}  // namespace forward_analysis
