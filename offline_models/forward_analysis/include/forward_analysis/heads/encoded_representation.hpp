#pragma once

#include "forward_analysis/domains/analysis_domain.hpp"
#include "forward_analysis/domains/analysis_domain_traits.hpp"

#include <Eigen/Core>

#include <algorithm>
#include <cstddef>

namespace forward_analysis {

// Shared Perceiver latent bottleneck Z (M x d_model) and domain-routed pooled vectors fed to NumericalPredictionHeads.
struct EncodedRepresentation {
    // Full Perceiver latent matrix Z (M x d_model).
    Eigen::MatrixXd latents_z;
    // Uniform mean-pooled latent vector (d_model).
    Eigen::VectorXd pooled_z;

    static EncodedRepresentation from_latents(const Eigen::MatrixXd& z) {
        // Constructed representation wrapper.
        EncodedRepresentation rep;
        rep.latents_z = z;
        if (z.rows() > 0) {
            rep.pooled_z = z.colwise().mean().transpose();
        }
        return rep;
    }

    // Pools Z with emphasis on the latent row assigned to `domain` (0.5 domain latent + 0.5 global mean).
    Eigen::VectorXd domain_vector(AnalysisDomain domain) const {
        if (latents_z.rows() == 0) {
            return pooled_z;
        }
        // Assigned latent index for `domain`.
        const Eigen::Index idx =
            static_cast<Eigen::Index>(AnalysisDomainTraits::index(domain) % static_cast<std::size_t>(latents_z.rows()));
        return 0.5 * pooled_z + 0.5 * latents_z.row(idx).transpose();
    }

    // distributes a gradient with respect to `domain_vector(domain)` back into `d_latents_z` (M x d_model).
    static void accumulate_domain_gradient(AnalysisDomain domain,
                                           const Eigen::VectorXd& d_domain_vec,
                                           Eigen::MatrixXd& d_latents_z) {
        if (d_latents_z.rows() == 0 || d_domain_vec.size() != d_latents_z.cols()) {
            return;
        }
        // Assigned latent index for `domain`.
        const Eigen::Index idx =
            static_cast<Eigen::Index>(AnalysisDomainTraits::index(domain) % static_cast<std::size_t>(d_latents_z.rows()));
        // Mean-pool gradient share per row.
        const double mean_scale = 0.5 / static_cast<double>(d_latents_z.rows());
        d_latents_z.rowwise() += d_domain_vec.transpose() * mean_scale;
        d_latents_z.row(idx) += 0.5 * d_domain_vec.transpose();
    }
};

}  // namespace forward_analysis
