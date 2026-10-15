#pragma once

#include "forward_analysis/heads/loss_result.hpp"
#include "forward_analysis/heads/prediction_result.hpp"

#include <Eigen/Core>

#include <map>
#include <string>

namespace forward_analysis {

struct AuxiliaryEvaluation {
    // Descriptor predicted from the latent bottleneck (same layout as ObservationDescriptor).
    Eigen::VectorXd predicted_observation_descriptor;
    // Decoder conditioning vector synthesized from the domain prediction heads (width 14).
    Eigen::VectorXd decoder_conditioning_vector;
    // Predicted dominant-eigenvector signs (+1 / -1).
    Eigen::VectorXd predicted_eigen_signs;
    // Predictions from all registered domain-specific NumericalPredictionHeads.
    std::map<std::string, PredictionResult> domain_predictions;
    // Losses and metrics from all registered domain-specific NumericalPredictionHeads.
    std::map<std::string, LossResult> domain_losses;
    // Predicted probability in [0, 1] that the network has two hidden layers (from LayerStructureHead).
    double predicted_two_layer_probability = 0.5;
    // Predicted probability in [0, 1] that the representation is narrow (from NarrowRepresentationHead).
    double predicted_narrow_probability = 0.5;
    // Predicted probability in [0, 1] that input-to-hidden association is nonzero (from FeatureAssociationHead).
    double predicted_nonzero_association_probability = 1.0;
    // Aux1: half mean squared error between predicted and measured descriptor.
    double aux1_observation_loss = 0.0;
    // Aux2: binary cross-entropy between predicted and target eigenvector signs.
    double aux2_eigen_sign_loss = 0.0;
};

}  // namespace forward_analysis
