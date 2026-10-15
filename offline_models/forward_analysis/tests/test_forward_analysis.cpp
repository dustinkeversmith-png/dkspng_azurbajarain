#include "forward_analysis/forward_analysis.hpp"

#include <chrono>
#include <cmath>
#include <cstddef>
#include <exception>
#include <filesystem>
#include <fstream>
#include <functional>
#include <iostream>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#ifndef FORWARD_ANALYSIS_DATA_DIR
#define FORWARD_ANALYSIS_DATA_DIR "data"
#endif
#ifndef FORWARD_ANALYSIS_FIXTURE_DIR
#define FORWARD_ANALYSIS_FIXTURE_DIR "tests/fixtures"
#endif

using namespace forward_analysis;

struct ForwardAnalysisTestSuite {
    // Number of require() checks that passed.
    std::size_t passed = 0;
    // Number of require() checks that failed.
    std::size_t failed = 0;
    // Model configuration shared by every check.
    DistillerConfig config;
    // Loaded training records (12, varying shapes).
    std::vector<TrainingSample> train;
    // Loaded held-out records (4, varying shapes).
    std::vector<TrainingSample> eval;

    void require(bool condition, const std::string& label) {
        if (condition) {
            ++passed;
        } else {
            ++failed;
            std::cout << "  FAIL " << label << "\n";
        }
    }

    void require_throws(const std::function<void()>& action, const std::string& label) {
        bool threw = false;
        try {
            action();
        } catch (const std::exception&) {
            threw = true;
        }
        require(threw, label);
    }

    static std::string data_path(const std::string& name) {
        return std::string(FORWARD_ANALYSIS_DATA_DIR) + "/" + name;
    }

    static std::string fixture_path(const std::string& name) {
        return std::string(FORWARD_ANALYSIS_FIXTURE_DIR) + "/" + name;
    }

    // The first `count` records of a JSONL file.
    static std::vector<TrainingSample> load_head(const TrainingDatasetImporter& importer, const std::string& path, std::size_t count) {
        std::ifstream in(path);
        std::string line;
        std::string head;
        for (std::size_t n = 0; n < count && std::getline(in, line); ++n) {
            head += line + "\n";
        }
        std::istringstream stream(head);
        return importer.read_jsonl(stream, path);
    }

    const TrainingSample& sample(const std::string& id) const {
        for (const std::vector<TrainingSample>* set : {&train, &eval}) {
            for (const TrainingSample& s : *set) {
                if (s.sample_id == id) {
                    return s;
                }
            }
        }
        throw std::runtime_error("missing sample " + id);
    }

    std::vector<const TrainingSample*> all_samples() const {
        std::vector<const TrainingSample*> out;
        for (const TrainingSample& s : train) {
            out.push_back(&s);
        }
        for (const TrainingSample& s : eval) {
            out.push_back(&s);
        }
        return out;
    }

    void verify_math() {
        std::cout << "math\n";
        Eigen::VectorXd a(4), b(4), ties(4);
        a << 1, 2, 3, 4;
        b << 2, 4, 6, 8;
        ties << 5, 1, 5, 2;
        require(std::abs(StatisticsOps::pearson(a, b) - 1.0) < 1e-12, "pearson of proportional vectors is 1");
        require(std::abs(StatisticsOps::pearson(a, Eigen::VectorXd::Constant(4, 3.0))) < 1e-12, "pearson with constant is 0");
        const Eigen::VectorXd r = StatisticsOps::ranks(ties);
        require(r(0) == 2.5 && r(2) == 2.5 && r(1) == 0.0 && r(3) == 1.0, "ranks average ties");
        require(StatisticsOps::same_order(a, b) && !StatisticsOps::same_order(a, -b), "same_order");

        Eigen::MatrixXd ray(3, 3);
        ray << 1, 2, 0, 2, 4, 0, 0.5, 1, 0;
        const Eigen::VectorXd s = SpectralOps::uncentered_singular_values(ray);
        require(s(1) < 1e-12 * s(0), "rank-1 matrix has one non-zero uncentered singular value");
        require(SpectralOps::principal_scores(ray, 2).rows() == 3 && SpectralOps::principal_scores(ray, 2).cols() == 2,
                "principal_scores is N x components");

        DeterministicRng rng(5);
        const Eigen::MatrixXd x = rng.xavier_matrix(3, 6) * 3.0;
        const Eigen::MatrixXd upstream = rng.xavier_matrix(3, 6);
        const Eigen::MatrixXd analytic = MatrixOps::layer_norm_rows_backward(
            MatrixOps::layer_norm_rows(x), MatrixOps::row_inverse_std(x), upstream);
        double worst = 0.0;
        for (Eigen::Index i = 0; i < x.rows(); ++i) {
            for (Eigen::Index j = 0; j < x.cols(); ++j) {
                Eigen::MatrixXd hi = x, lo = x;
                hi(i, j) += 1e-6;
                lo(i, j) -= 1e-6;
                const double numeric =
                    ((MatrixOps::layer_norm_rows(hi) - MatrixOps::layer_norm_rows(lo)).cwiseProduct(upstream)).sum() / 2e-6;
                worst = std::max(worst, std::abs(numeric - analytic(i, j)));
            }
        }
        require(worst < 1e-5, "layer_norm_rows_backward matches finite differences");

        AdamMoment moment;
        Eigen::MatrixXd param = Eigen::MatrixXd::Constant(2, 2, 3.0);
        for (std::uint64_t step = 1; step <= 400; ++step) {
            moment.apply(param, param, 0.05, step);
        }
        require(param.cwiseAbs().maxCoeff() < 0.5, "Adam minimizes a quadratic");
        Eigen::MatrixXd grown = Eigen::MatrixXd::Ones(3, 2);
        moment.apply(grown, grown, 0.01, 401);
        require(grown.rows() == 3, "AdamMoment grows with its parameter");
    }

    void verify_text_helpers() {
        std::cout << "text helpers\n";
        std::size_t v = 0;
        require(NumberWords::word(24) == "twenty-four" && NumberWords::parse("Twenty-four", v) && v == 24, "NumberWords");
        require(EnglishListFormatter::coordinates({0, 5, 12}) == "x0, x5 and x12", "EnglishListFormatter");
        const std::string text = "Units 3, 8 and 14 (of twenty-four) follow x0/x7; the rest: none.";
        require(LexemeSplitter::join(LexemeSplitter::split(text)) == text, "LexemeSplitter round trip");
        ObservationText same;
        same.pathway_activity = same.feature_sensitivity = same.representational_structure = same.functional_mapping = "a b c";
        ObservationText other = same;
        other.pathway_activity = "d e";
        require(std::abs(ObservationSimilarity::mean_f1(same, same) - 1.0) < 1e-12, "identical text has F1 1");
        require(ObservationSimilarity::field_f1("a b c", "d e") == 0.0, "disjoint text has F1 0");
        require(std::abs(ObservationSimilarity::mean_f1(other, same) - 0.75) < 1e-12, "mean F1 averages fields");
    }

    void verify_builder() {
        std::cout << "telemetry builder\n";
        const Eigen::MatrixXd in = Eigen::MatrixXd::Ones(4, 3);
        const Eigen::MatrixXd out = Eigen::MatrixXd::Ones(4, 2);
        const Eigen::MatrixXd hidden = Eigen::MatrixXd::Ones(4, 5);
        const ForwardTelemetryTensor t = TelemetryBuilder::build(in, out, out * 0.5, {hidden});
        require(t.shape().to_string() == "N=4 in=3 hidden=[5] out=2", "TelemetryShape string");
        require(t.shape().channel_count() == 3 + 5 + 3 * 2, "channel count = inputs + hidden units + 3 x outputs");
        require(t.residuals.isApprox(-0.5 * out), "residuals derived");
        require_throws([&] { TelemetryBuilder::build(in, Eigen::MatrixXd::Ones(3, 2), out, {hidden}); }, "row mismatch rejected");
        require_throws([&] { TelemetryBuilder::build(in, out, Eigen::MatrixXd::Ones(4, 3), {hidden}); }, "width mismatch rejected");
        require_throws([&] { TelemetryBuilder::build(in, out, out, {}); }, "missing hidden layer rejected");
        require_throws([&] { TelemetryBuilder::build(in, out, out, {Eigen::MatrixXd(4, 0)}); }, "zero-unit layer rejected");
        Eigen::MatrixXd bad = in;
        bad(0, 0) = std::nan("");
        require_throws([&] { TelemetryBuilder::build(bad, out, out, {hidden}); }, "non-finite rejected");
    }

    void verify_importer() {
        std::cout << "dataset importer\n";
        const TrainingDatasetImporter importer(config);
        train = importer.load_jsonl(fixture_path("llm_train.jsonl"));
        eval = importer.load_jsonl(fixture_path("llm_eval.jsonl"));
        require(train.size() == 40 && eval.size() == 10, "40 train and 10 eval records");
        std::set<std::size_t> input_dims;
        std::set<std::size_t> layer_counts;
        std::set<std::size_t> output_dims;
        for (const TrainingSample* s : all_samples()) {
            input_dims.insert(s->telemetry.shape().input_dim);
            layer_counts.insert(s->telemetry.num_layers());
            output_dims.insert(s->telemetry.shape().output_dim);
            require(s->ground_truth_targets.eigen_signs.size() == 4, s->sample_id + " has four eigen signs");
            for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
                require(!s->ground_truth_targets.observations.field(f).empty(), s->sample_id + " field non-empty");
            }
        }
        require(input_dims.size() >= 5 && layer_counts.size() == 2 && output_dims.size() == 4, "shapes vary across records");
        const DatasetShapeSummary summary = DatasetShapeSummary::from(train);
        require(summary.min_input_dim == 6 && summary.max_input_dim == 16, "summary input range 6-16");

        const std::string good = R"({"sample_id":"s","telemetry":{"inputs":[[1,2],[3,4]],"targets":[[1],[2]],"predictions":[[1],[1]],"layer_activations":[[[0,1],[0,2]]]},"ground_truth_targets":{"target_text":{"pathway_activity":"a","feature_sensitivity":"b","representational_structure":"c","functional_mapping":"d"}},"extra":1})";
        std::istringstream one(good + "\n");
        const std::vector<TrainingSample> parsed = importer.read_jsonl(one, "inline");
        require(parsed.size() == 1 && parsed.front().ground_truth_targets.eigen_signs.size() == 4,
                "missing eigen_signs fall back to measured, unknown keys ignored");

        std::istringstream broken(good + "\n{not json}\n");
        bool located = false;
        try {
            importer.read_jsonl(broken, "inline");
        } catch (const std::exception& e) {
            located = std::string(e.what()).rfind("inline:2:", 0) == 0;
        }
        require(located, "parse errors report source:line");
        std::string no_field = good;
        no_field.replace(no_field.find("\"functional_mapping\":\"d\""), 24, "\"other\":\"d\"");
        std::istringstream missing(no_field);
        require_throws([&] { importer.read_jsonl(missing, "inline"); }, "missing observation field rejected");
        std::istringstream empty("\n\n");
        require_throws([&] { importer.read_jsonl(empty, "inline"); }, "empty dataset rejected");
    }

    void verify_extractor() {
        std::cout << "observation extractor\n";
        const NetworkObservations& t1 = sample("syn_math_train_001").measured;
        require(t1.layers[0].dead_units == std::vector<std::size_t>({3, 8, 14, 19, 22}), "train_001 dead units");
        require(t1.layers[0].lies_on_ray && t1.layers[1].lies_on_ray, "train_001 both layers on a ray");
        require(t1.outputs_move_in_lockstep, "train_001 outputs in lockstep");

        const NetworkObservations& t3 = sample("syn_math_train_003").measured;
        require(t3.gating.present() && t3.gating.has_gate_coordinate && t3.gating.gate_coordinate == 3, "train_003 gated by x3");
        require(t3.gating.closed_samples == std::vector<std::size_t>({1, 4, 5}), "train_003 closed samples");
        require(t3.gating.deeper_layers_constant, "train_003 deeper layer constant on closed samples");

        const NetworkObservations& t4 = sample("syn_math_train_004").measured;
        require(t4.outputs[0].sign_preserved && t4.outputs[0].order_preserved && t4.outputs[0].amplitude_ratio < 0.7,
                "train_004 output keeps sign and order but is damped");

        require(sample("syn_math_train_002").measured.layers[0].dead_units.size() == 11, "train_002 eleven dead units");
        const NetworkObservations& e4 = sample("syn_math_eval_004").measured;
        require(e4.layers[0].lies_on_ray && e4.layers[0].dead_units.size() == 7, "eval_004 ray with seven dead units");

        bool found_x10 = false;
        for (const InputDriver& d : sample("syn_math_train_010").measured.drivers) {
            found_x10 = found_x10 || d.coordinate == 10;
        }
        require(found_x10, "train_010 x10 drives the hidden plane");

        for (const TrainingSample* s : all_samples()) {
            require(s->measured.eigen_signs.size() == 4 && s->measured.eigen_signs.cwiseAbs().isApprox(Eigen::VectorXd::Ones(4)),
                    s->sample_id + " four measured eigen signs of +/-1");
            const Eigen::VectorXd d = ObservationDescriptor::encode(s->measured);
            require(d.size() == static_cast<Eigen::Index>(ObservationDescriptor::dim()) && d.allFinite(),
                    s->sample_id + " descriptor finite");
        }
    }

    // Measurement, checker and narrator against the corrected dataset (manifest corrected-2026-10-14).
    void verify_corrected_dataset() {
        std::cout << "corrected dataset measurements\n";
        Eigen::MatrixXd inputs(4, 4);
        inputs << 1, 2, 0, 1,
                  2, 1, 1, 0,
                  3, 4, 0, 2,
                  4, 3, 1, 1;
        Eigen::MatrixXd centered = inputs.rowwise() - inputs.colwise().mean();
        Eigen::MatrixXd covariance = centered.transpose() * centered / 3.0;
        covariance.diagonal().setZero();
        const Eigen::VectorXd eigenvalues = Eigen::SelfAdjointEigenSolver<Eigen::MatrixXd>(covariance).eigenvalues().reverse();
        const Eigen::VectorXd expected = eigenvalues.unaryExpr([](double v) { return v >= 0.0 ? 1.0 : -1.0; });
        require(ObservationExtractor::eigen_signs(inputs, config).isApprox(expected),
                "eigen signs are the descending eigenvalue signs of the zero-diagonal input covariance");
        require(ObservationExtractor::eigen_signs(inputs.leftCols(2), config).tail(2).isApprox(Eigen::VectorXd::Ones(2)),
                "eigen signs beyond the input width stay +1");

        const TrainingDatasetImporter importer(config);
        for (const char* file : {"llm_train.jsonl", "llm_validation.jsonl", "llm_eval_ood.jsonl"}) {
            const std::string path = data_path(file);
            if (!std::filesystem::exists(path)) {
                std::cout << "  " << file << " not present, skipped\n";
                continue;
            }
            const std::vector<TrainingSample> head = load_head(importer, path, 200);
            std::size_t sign_match = 0;
            std::size_t truth_pass = 0;
            std::size_t narrated_exact = 0;
            // Fields whose batch-statistic slot lexemes relexicalize to the ground-truth lexemes.
            std::size_t slot_round_trips = 0;
            for (const TrainingSample& s : head) {
                sign_match += s.ground_truth_targets.eigen_signs.isApprox(s.measured.eigen_signs) ? 1 : 0;
                const ObservationText narrated = BatchStatisticsNarrator::describe(s.measured, config);
                for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
                    const std::string& truth = s.ground_truth_targets.observations.field(f);
                    truth_pass += ObservationConsistency::verify_field(truth, s.measured) ? 1 : 0;
                    narrated_exact += narrated.field(f) == truth ? 1 : 0;
                    // Ground-truth field with every batch-statistic number replaced by its slot.
                    const std::vector<std::string> slots =
                        FieldDelexicalizer::delexicalize(truth, s.measured, SlotBinding::kBatchStatistics);
                    // Slots filled back from the measured telemetry.
                    std::string refilled;
                    slot_round_trips += ObservationDelexicalizer::relexicalize(slots, ShapeSlotLexicon::from(s.measured), refilled) &&
                                                LexemeSplitter::split(refilled) == LexemeSplitter::split(truth)
                                            ? 1
                                            : 0;
                    require(ObservationConsistency::verify_field(narrated.field(f), s.measured),
                            s.sample_id + " batch narration " + ObservationText::field_names()[f] + " consistent");
                }
            }
            std::cout << "  " << file << " first " << head.size() << ": eigen signs " << sign_match << ", ground-truth fields consistent "
                      << truth_pass << "/" << head.size() * 4 << ", narration identical to ground truth " << narrated_exact
                      << ", batch slots round-trip " << slot_round_trips << "\n";
            require(slot_round_trips + 4 >= head.size() * 4, std::string(file) + " batch-statistic slots refill to the ground truth");
            require(sign_match == head.size(), std::string(file) + " file eigen signs equal measured eigen signs");
            require(truth_pass + 4 >= head.size() * 4, std::string(file) + " ground truth passes the consistency check");
            require(narrated_exact + 4 >= head.size() * 4, std::string(file) + " batch narration reproduces the ground truth");
        }

        const TrainingSample& t4 = sample("syn_math_train_004");
        const NetworkObservations& m = t4.measured;
        const std::string truth = BatchStatisticsNarrator::describe(m, config).pathway_activity;
        require(ObservationConsistency::verify_field(truth, m), "batch pathway narration consistent");
        require(!ObservationConsistency::verify_field("Across 99 observations, the hidden widths are 3.", m),
                "wrong observation count rejected");
        require(!ObservationConsistency::verify_field(
                    "Layer 1 has " + std::to_string(m.layers[0].dead_units.size() + 1) + " coordinates that remain zero throughout the batch.", m),
                "wrong always-zero count rejected");
        require(!ObservationConsistency::verify_field("Layer 7 has 0 coordinates that remain zero throughout the batch.", m),
                "missing layer rejected");
        require(!ObservationConsistency::verify_field(
                    "Prediction and target signs agree in " + std::to_string(m.sign_agreements + 1) + " of " +
                        std::to_string(m.sign_entries) + " entries.", m),
                "wrong sign agreement rejected");
        require(ObservationConsistency::verify_field(
                    "Prediction and target signs agree in " + std::to_string(m.sign_agreements) + " of " +
                        std::to_string(m.sign_entries) + " entries.", m),
                "measured sign agreement accepted even when the counts exceed every layer width");
        require(!ObservationConsistency::verify_field(m.narrow_representation ? "The observed activation variation occupies multiple directions."
                                                                              : "This is a narrow observed representation.", m),
                "narrow / multiple-direction claim checked against numerical ranks");

        // Second-layer width slot, unfillable for a one-layer network.
        const std::vector<std::string> second_width =
            BatchStatementDelexicalizer::delexicalize("Across 5 observations, the hidden widths are 8 and 4.");
        require(second_width.size() > 2 && second_width[second_width.size() - 2] == "<COUNT:W_2|d>",
                "hidden-width list delexicalized positionally");
        // One-layer copy of the sample-004 observations.
        NetworkObservations one_layer = m;
        one_layer.layers.resize(1);
        // Text relexicalized for the one-layer observations.
        std::string unfilled;
        require(!ObservationDelexicalizer::relexicalize(second_width, ShapeSlotLexicon::from(one_layer), unfilled),
                "second-layer slot not fillable for a one-layer network");
    }

    void verify_encoder() {
        std::cout << "channel encoder\n";
        ChannelSequenceFormatter formatter(config);
        const PerceiverIOEncoder perceiver(config);
        for (const TrainingSample* s : all_samples()) {
            const Eigen::MatrixXd features = formatter.channel_features(s->telemetry);
            require(features.rows() == static_cast<Eigen::Index>(s->telemetry.shape().channel_count()) &&
                        features.cols() == static_cast<Eigen::Index>(ChannelFeatureExtractor::kFeatureDim) &&
                        features.allFinite() && features.cwiseAbs().maxCoeff() <= 1.0 + 1e-9,
                    s->sample_id + " channel features C x F, bounded");
            const Eigen::MatrixXd z = perceiver.encode(formatter.format_sequence(features));
            require(z.rows() == static_cast<Eigen::Index>(config.num_latents) &&
                        z.cols() == static_cast<Eigen::Index>(config.d_model) && z.allFinite(),
                    s->sample_id + " Z is M x d regardless of shape");
        }

        const TrainingSample& s = sample("syn_math_train_004");
        ForwardTelemetryTensor scaled = s.telemetry;
        Eigen::MatrixXd permuted_inputs = scaled.inputs;
        permuted_inputs.col(0).swap(permuted_inputs.col(1));
        scaled.inputs = permuted_inputs;
        const Eigen::MatrixXd original = perceiver.encode(formatter.format_sequence(s.telemetry));
        const Eigen::MatrixXd swapped = perceiver.encode(formatter.format_sequence(scaled));
        require((original - swapped).cwiseAbs().maxCoeff() < 0.5, "swapping two input columns changes Z only through index features");
    }

    void verify_domain_feature_extractors() {
        std::cout << "domain feature extractors and feature significance\n";
        // Registry initialized with the six default mathematical domain extractors.
        FeatureExtractorRegistry registry;
        require(registry.size() == AnalysisDomainTraits::kDomainCount, "all 6 analysis domains registered by default");
        // Verify descriptors on an existing training sample.
        const TrainingSample& t4 = sample("syn_math_train_004");
        for (AnalysisDomain domain : AnalysisDomainTraits::all()) {
            require(!registry.extract_domain(domain, t4.telemetry, config).empty(),
                    std::string("extractor present for ") + AnalysisDomainTraits::name(domain));
        }

        // Extracted feature descriptors across all domains.
        const std::vector<FeatureDescriptor> t4_features = registry.extract_all(t4.telemetry, config);
        require(t4_features.size() >= 10 && registry.flatten_values(t4.telemetry, config).size() >= 50,
                "default registry extracts multi-domain feature descriptors");
        require(registry.flatten_values(t4.telemetry, config).allFinite(), "flattened feature values finite");

        // Verify differentiable normalization derivative against finite differences.
        for (FeatureNormalizationStrategy strat :
             {FeatureNormalizationStrategy::Identity,
              FeatureNormalizationStrategy::UnitInterval,
              FeatureNormalizationStrategy::SignedUnitInterval,
              FeatureNormalizationStrategy::TanhSquash,
              FeatureNormalizationStrategy::LogScale,
              FeatureNormalizationStrategy::StandardScore}) {
            // Probe scalar value.
            const double x = 0.42;
            // Finite difference step.
            const double eps = 1e-6;
            // Numeric derivative.
            const double num_d =
                (FeatureDescriptor::normalize_value(x + eps, strat) -
                 FeatureDescriptor::normalize_value(x - eps, strat)) /
                (2.0 * eps);
            // Analytic derivative.
            const double ana_d = FeatureDescriptor::normalize_derivative(x, strat);
            require(std::abs(num_d - ana_d) < 1e-5, "FeatureDescriptor::normalize_derivative matches finite difference");
        }

        // Verify extensibility: register a custom extractor without modifying PerceiverIOEncoder.
        registry.register_extractor(std::make_shared<CustomProbeExtractor>());
        require(registry.size() == AnalysisDomainTraits::kDomainCount + 1, "custom extractor registered dynamically");

        // --- Feature Significance Distinction Checks (Section 4) ---
        // 1. Narrow vs multidirectional representations:
        // Base input matrix (8 samples x 4 coordinates).
        Eigen::MatrixXd inputs_base(8, 4);
        inputs_base << -1.5, 0.5, -0.5, 1.0,
                       -1.0, -0.5, 0.8, -0.8,
                       -0.5, 1.2, -1.1, 0.3,
                        0.0, -1.0, 0.4, -0.4,
                        0.5, 0.8, -0.9, 0.9,
                        1.0, -0.7, 1.2, -1.1,
                        1.2, 0.3, -0.3, 0.2,
                       -1.2, -0.6, 0.4, -0.1;
        // Rank-1 (narrow) hidden activations.
        Eigen::MatrixXd h_narrow = inputs_base.col(0) * Eigen::RowVector4d(1.0, 2.0, 0.5, 1.5);
        // Full-rank (multidirectional) hidden activations.
        Eigen::MatrixXd h_multi = inputs_base.cwiseMax(0.0);
        // Narrow telemetry tensor.
        ForwardTelemetryTensor tel_narrow =
            TelemetryBuilder::build(inputs_base, inputs_base.col(0), inputs_base.col(0), {h_narrow});
        // Multidirectional telemetry tensor.
        ForwardTelemetryTensor tel_multi =
            TelemetryBuilder::build(inputs_base, inputs_base.col(0), inputs_base.col(0), {h_multi});
        // Geometric domain extractor instance.
        GeometricFeatureExtractor geom_ext;
        // Extracted geometric features for narrow telemetry.
        const std::vector<FeatureDescriptor> f_narrow = geom_ext.extract(tel_narrow, config);
        // Extracted geometric features for multidirectional telemetry.
        const std::vector<FeatureDescriptor> f_multi = geom_ext.extract(tel_multi, config);
        require(f_narrow[0].values[0] == 1.0 && f_multi[0].values[0] == 0.0,
                "significance 1: narrow vs multidirectional representation distinguished");

        // 2. Positive vs negative spectral modes (+++- vs +--- on 4x4 zero-diagonal input covariance):
        // Orthogonal zero-mean 8x4 basis matrix.
        Eigen::MatrixXd u_ortho(8, 4);
        u_ortho <<  1,  1,  1,  1,
                    1,  1, -1, -1,
                    1, -1,  1, -1,
                    1, -1, -1,  1,
                   -1,  1,  1, -1,
                   -1,  1, -1,  1,
                   -1, -1,  1,  1,
                   -1, -1, -1, -1;
        // Mixing matrix producing all-positive off-diagonal covariance (+---).
        Eigen::Matrix4d m_pos = Eigen::Matrix4d::Constant(0.45);
        m_pos.diagonal().setConstant(0.85);
        // Mixing matrix producing all-negative off-diagonal covariance (+++-).
        Eigen::Matrix4d m_neg = Eigen::Matrix4d::Constant(-0.25);
        m_neg.diagonal().setConstant(0.95);
        // Telemetry with +--- spectral mode.
        ForwardTelemetryTensor tel_spec_pos =
            TelemetryBuilder::build(u_ortho * m_pos, inputs_base.col(0), inputs_base.col(0), {h_multi});
        // Telemetry with +++- spectral mode.
        ForwardTelemetryTensor tel_spec_neg =
            TelemetryBuilder::build(u_ortho * m_neg, inputs_base.col(0), inputs_base.col(0), {h_multi});
        // Spectral domain extractor instance.
        SpectralFeatureExtractor spec_ext;
        // Spectral descriptors for +--- mode.
        const std::vector<FeatureDescriptor> s_pos = spec_ext.extract(tel_spec_pos, config);
        // Spectral descriptors for +++- mode.
        const std::vector<FeatureDescriptor> s_neg = spec_ext.extract(tel_spec_neg, config);
        require(s_pos[1].values[1] < 0.0 && s_neg[1].values[1] > 0.0 &&
                    s_pos[1].values[2] < 0.0 && s_neg[1].values[2] > 0.0,
                "significance 2: positive vs negative spectral modes (+--- vs +++-) distinguished");

        // 3. One-layer vs multilayer architectures:
        // Two-layer telemetry tensor.
        ForwardTelemetryTensor tel_2layer =
            TelemetryBuilder::build(inputs_base, inputs_base.col(0), inputs_base.col(0), {h_multi, h_narrow});
        // Structural domain extractor instance.
        StructuralTopologyFeatureExtractor struct_ext;
        // Structural descriptors for 1-layer network.
        const std::vector<FeatureDescriptor> st_1 = struct_ext.extract(tel_multi, config);
        // Structural descriptors for 2-layer network.
        const std::vector<FeatureDescriptor> st_2 = struct_ext.extract(tel_2layer, config);
        require(st_1[0].values[0] == 1.0 && st_2[0].values[1] == 1.0 && !st_1[1].valid && st_2[1].valid,
                "significance 3: 1-layer vs 2-layer architecture and missing-layer validity distinguished");

        // 4. Similar moments but different sample orderings:
        // Input matrix with monotonically sorted first column.
        Eigen::MatrixXd in_sorted = inputs_base;
        in_sorted.col(0) << -3.0, -2.0, -1.0, -0.5, 0.5, 1.0, 2.0, 3.0;
        // Input matrix with alternating first column of identical mean and variance.
        Eigen::MatrixXd in_alternating = inputs_base;
        in_alternating.col(0) << -3.0, 3.0, -2.0, 2.0, -1.0, 1.0, -0.5, 0.5;
        // Telemetry for sorted sample ordering.
        ForwardTelemetryTensor tel_sorted =
            TelemetryBuilder::build(in_sorted, in_sorted.col(0), in_sorted.col(0), {h_multi});
        // Telemetry for alternating sample ordering.
        ForwardTelemetryTensor tel_alt =
            TelemetryBuilder::build(in_alternating, in_alternating.col(0), in_alternating.col(0), {h_multi});
        // Statistical domain extractor instance.
        StatisticalFeatureExtractor stat_ext;
        // Statistical descriptors for sorted ordering.
        const std::vector<FeatureDescriptor> stat_sorted = stat_ext.extract(tel_sorted, config);
        // Statistical descriptors for alternating ordering.
        const std::vector<FeatureDescriptor> stat_alt = stat_ext.extract(tel_alt, config);
        require(std::abs(stat_sorted[0].values[0] - stat_alt[0].values[0]) < 1e-12 &&
                    std::abs(stat_sorted[0].values[1] - stat_alt[0].values[1]) < 1e-12 &&
                    std::abs(stat_sorted[1].values[0] - stat_alt[1].values[0]) > 0.3,
                "significance 4: identical marginal moments with different sample orderings distinguished");

        // 5. Different correlation structures with identical marginal statistics:
        // Positively correlated first two input columns.
        Eigen::MatrixXd in_corr = in_sorted;
        in_corr.col(1) = in_sorted.col(0);
        // Negatively correlated first two input columns with identical marginal distribution.
        Eigen::MatrixXd in_anticorr = in_sorted;
        in_anticorr.col(1) = -in_sorted.col(0);
        // Telemetry for positively correlated inputs.
        ForwardTelemetryTensor tel_corr =
            TelemetryBuilder::build(in_corr, in_corr.col(0), in_corr.col(0), {h_multi});
        // Telemetry for negatively correlated inputs.
        ForwardTelemetryTensor tel_anticorr =
            TelemetryBuilder::build(in_anticorr, in_anticorr.col(0), in_anticorr.col(0), {h_multi});
        // Statistical descriptors for correlated inputs.
        const std::vector<FeatureDescriptor> stat_corr = stat_ext.extract(tel_corr, config);
        // Statistical descriptors for anticorrelated inputs.
        const std::vector<FeatureDescriptor> stat_anticorr = stat_ext.extract(tel_anticorr, config);
        require(std::abs(stat_corr[2].values[1] - stat_anticorr[2].values[1]) > 0.25,
                "significance 5: different correlation structures with identical marginals distinguished");

        // 6. Different output error structures:
        // Telemetry with aligned predictions and targets.
        ForwardTelemetryTensor tel_aligned =
            TelemetryBuilder::build(in_sorted, in_sorted.col(0), in_sorted.col(0), {h_multi});
        // Telemetry with sign-inverted predictions relative to targets.
        ForwardTelemetryTensor tel_flipped =
            TelemetryBuilder::build(in_sorted, -in_sorted.col(0), in_sorted.col(0), {h_multi});
        // Functional domain extractor instance.
        FunctionalMappingFeatureExtractor func_ext;
        // Functional descriptors for aligned predictions.
        const std::vector<FeatureDescriptor> fn_aligned = func_ext.extract(tel_aligned, config);
        // Functional descriptors for sign-inverted predictions.
        const std::vector<FeatureDescriptor> fn_flipped = func_ext.extract(tel_flipped, config);
        require(fn_aligned[0].values[3] > 0.9 && fn_flipped[0].values[3] < 0.1 &&
                    fn_aligned[0].values[6] > 0.9 && fn_flipped[0].values[6] < -0.9,
                "significance 6: aligned vs sign-flipped output error structures distinguished");
    }

    void verify_multi_family_tokens() {
        std::cout << "multi-family token representations (5 ablation levels)\n";
        // One-layer reference sample.
        const TrainingSample& t4 = sample("syn_math_train_004");
        // Two-layer reference sample.
        const TrainingSample& t1 = sample("syn_math_train_001");
        // Token count from the previous ablation level.
        std::size_t prev_tokens = 0;
        for (FeatureAblationLevel level :
             {FeatureAblationLevel::kBaseline25,
              FeatureAblationLevel::kExpandedChannel,
              FeatureAblationLevel::kChannelPlusSample,
              FeatureAblationLevel::kChannelPlusRelationship,
              FeatureAblationLevel::kCompleteArchitecture}) {
            // Config with the current ablation level.
            DistillerConfig cfg = config;
            cfg.feature_ablation_level = level;
            // Formatter for the current ablation level.
            ChannelSequenceFormatter formatter(cfg);
            // Perceiver encoder for the current ablation level.
            PerceiverIOEncoder perceiver(cfg);
            // Multi-family token batch for 1-layer sample.
            const MultiFamilyTokenBatch b1 = formatter.extract_token_batch(t4.telemetry);
            // Multi-family token batch for 2-layer sample.
            const MultiFamilyTokenBatch b2 = formatter.extract_token_batch(t1.telemetry);
            require(b1.total_tokens() >= prev_tokens, "token count grows or stays equal with higher ablation level");
            prev_tokens = b1.total_tokens();
            // Projected token sequence (T x d_model).
            const Eigen::MatrixXd seq1 = formatter.format_sequence(b1);
            // Encoded Perceiver latent matrix (M x d_model).
            const Eigen::MatrixXd z1 = perceiver.encode(seq1);
            require(seq1.rows() == static_cast<Eigen::Index>(b1.total_tokens()) &&
                        seq1.cols() == static_cast<Eigen::Index>(cfg.d_model) && z1.allFinite(),
                    "formatted multi-family sequence and Perceiver latents valid");
            if (level == FeatureAblationLevel::kCompleteArchitecture) {
                require(b1.layer_valid_mask.size() == 2 && b1.layer_valid_mask[0] == 1.0 && b1.layer_valid_mask[1] == 0.0,
                        "1-layer network masks out Layer 2 token");
                require(b2.layer_valid_mask.size() == 2 && b2.layer_valid_mask[0] == 1.0 && b2.layer_valid_mask[1] == 1.0,
                        "2-layer network keeps both Layer 1 and Layer 2 tokens active");
            }
        }
    }

    void verify_domain_prediction_heads() {
        std::cout << "typed domain prediction heads and eigen-sign learning\n";
        // Default domain prediction head registry.
        NumericalPredictionHeadRegistry head_reg(config, 1234ULL);
        require(head_reg.size() == 8, "8 domain-specific prediction heads registered");

        // Verify missing-label masking on 1-layer vs 2-layer samples.
        // One-layer training sample.
        const TrainingSample& one_layer_sample = sample("syn_math_train_004");
        // Two-layer training sample.
        const TrainingSample& two_layer_sample = sample("syn_math_train_001");
        // Targets for 1-layer sample.
        const NumericalTargets targets_1 = NumericalTargets::from_observations(
            one_layer_sample.measured,
            one_layer_sample.ground_truth_targets.eigen_signs,
            config);
        // Targets for 2-layer sample.
        const NumericalTargets targets_2 = NumericalTargets::from_observations(
            two_layer_sample.measured,
            two_layer_sample.ground_truth_targets.eigen_signs,
            config);
        require(targets_1.masks.at("activation_sparsity")(3) == 0.0 &&
                    targets_1.masks.at("numerical_rank")(2) == 0.0 &&
                    targets_1.masks.at("narrow_representation")(2) == 0.0 &&
                    targets_1.masks.at("layer_structure")(2) == 0.0,
                "1-layer sample masks Layer 2 targets in sparsity, rank, narrow, and layer_structure heads");
        require(targets_2.masks.at("activation_sparsity")(3) == 1.0 &&
                    targets_2.masks.at("numerical_rank")(2) == 1.0 &&
                    targets_2.masks.at("narrow_representation")(2) == 1.0 &&
                    targets_2.masks.at("layer_structure")(2) == 1.0,
                "2-layer sample unmasks Layer 2 targets");

        // Verify EigenSignHead learns non-majority spectral sign patterns (+++- and +---) as well as ++--.
        // Redesigned EigenSignHead instance.
        EigenSignHead eigen_head(config.d_model, config.eigen_sign_dim, 777ULL);
        // RNG for synthetic latent representations.
        DeterministicRng rng(777ULL);
        // Pooled latent representation for +++- pattern.
        const Eigen::VectorXd z_pppm = random_matrix(rng, static_cast<Eigen::Index>(config.d_model), 1).col(0);
        // Pooled latent representation for +--- pattern.
        const Eigen::VectorXd z_pmmm = random_matrix(rng, static_cast<Eigen::Index>(config.d_model), 1).col(0);
        // Pooled latent representation for ++-- pattern.
        const Eigen::VectorXd z_ppmm = random_matrix(rng, static_cast<Eigen::Index>(config.d_model), 1).col(0);
        // Target signs +++-.
        Eigen::Vector4d y_pppm(1.0, 1.0, 1.0, -1.0);
        // Target signs +---.
        Eigen::Vector4d y_pmmm(1.0, -1.0, -1.0, -1.0);
        // Target signs ++--.
        Eigen::Vector4d y_ppmm(1.0, 1.0, -1.0, -1.0);
        // Optimization step counter.
        for (int step = 0; step < 60; ++step) {
            // Gradient accumulator wrt pooled_z.
            Eigen::VectorXd dz = Eigen::VectorXd::Zero(static_cast<Eigen::Index>(config.d_model));
            eigen_head.compute_loss_and_step(z_pppm, y_pppm, 0.05, 1.0, dz);
            eigen_head.compute_loss_and_step(z_pmmm, y_pmmm, 0.05, 1.0, dz);
            eigen_head.compute_loss_and_step(z_ppmm, y_ppmm, 0.05, 1.0, dz);
        }
        require((eigen_head.predict_signs(z_pppm) - y_pppm).norm() == 0.0 &&
                    (eigen_head.predict_signs(z_pmmm) - y_pmmm).norm() == 0.0 &&
                    (eigen_head.predict_signs(z_ppmm) - y_ppmm).norm() == 0.0,
                "redesigned EigenSignHead accurately classifies +++-, +---, and ++-- without majority collapse");
    }

    void verify_description() {
        std::cout << "narrator and consistency\n";
        std::size_t truth_pass = 0;
        std::size_t truth_total = 0;
        for (const TrainingSample* s : all_samples()) {
            const ObservationText narrated = ObservationNarrator::describe(s->measured);
            for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
                const std::string reason = ObservationConsistency::violation(narrated.field(f), s->measured);
                require(!narrated.field(f).empty() && reason.empty(),
                        s->sample_id + " narrated " + ObservationText::field_names()[f] + " consistent " + reason);
                truth_pass += ObservationConsistency::verify_field(s->ground_truth_targets.observations.field(f), s->measured);
                ++truth_total;
            }
        }
        std::cout << "  ground-truth fields passing the consistency check: " << truth_pass << "/" << truth_total << "\n";

        const NetworkObservations& t4 = sample("syn_math_train_004").measured;
        require(!ObservationConsistency::verify_field("x99 drives everything.", t4), "out-of-range coordinate rejected");
        require(!ObservationConsistency::verify_field("The second layer is flat.", t4), "second layer rejected on one-layer network");
        require(!ObservationConsistency::verify_field("The sign of x2 gates the layer.", t4), "gating rejected on ungated network");
        require(!ObservationConsistency::verify_field("Three of the ten units are dead.", t4), "wrong count pair rejected");
        require(!ObservationConsistency::verify_field("a b c d a b c d a b c d", t4), "repetition loop rejected");
        require(ObservationConsistency::verify_field("No unit is dead.", t4), "negated dead claim accepted");
        require(!ObservationConsistency::verify_field("The prediction is flat.", t4), "flat rejected when the output varies");
        require(!ObservationConsistency::verify_field("The prediction rises.", sample("syn_math_train_001").measured),
                "singular prediction rejected on a four-output network");

        ObservationTokenizer tokenizer;
        std::size_t round_trips = 0;
        for (const TrainingSample& s : train) {
            for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
                const std::string& text = s.ground_truth_targets.observations.field(f);
                round_trips += tokenizer.decode(tokenizer.encode(text, 512, true)) == text ? 1 : 0;
            }
        }
        require(round_trips == train.size() * ObservationText::kFieldCount, "tokenizer round-trips every ground-truth field");
        require(tokenizer.encode_known("zzzunseen", 8)[1] == ObservationTokenizer::kUnkToken, "unseen word maps to UNK");
    }

    static std::vector<std::size_t> coordinate_mentions(const std::string& text) {
        std::vector<std::size_t> out;
        for (const std::string& lexeme : LexemeSplitter::split(text)) {
            const std::string w = LexemeSplitter::lowercase(lexeme);
            if (w.size() > 1 && w[0] == 'x' && w.find_first_not_of("0123456789", 1) == std::string::npos) {
                out.push_back(static_cast<std::size_t>(std::stoul(w.substr(1))));
            }
        }
        return out;
    }

    void verify_slots() {
        std::cout << "slots (improvement 1)\n";
        std::size_t round_trips = 0;
        std::size_t slot_count = 0;
        std::size_t literal_coordinates = 0;
        std::size_t literal_numbers_ge2 = 0;
        for (const TrainingSample* s : all_samples()) {
            const ShapeSlotLexicon lex = ShapeSlotLexicon::from(s->measured);
            for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
                const std::string& text = s->ground_truth_targets.observations.field(f);
                const std::vector<std::string> slotted = ObservationDelexicalizer::delexicalize(text, lex);
                std::string filled;
                const bool ok = ObservationDelexicalizer::relexicalize(slotted, lex, filled);
                round_trips += ok && filled == text ? 1 : 0;
                if (!(ok && filled == text)) {
                    std::cout << "  round trip differs " << s->sample_id << "." << ObservationText::field_names()[f] << "\n    "
                              << text << "\n    " << filled << "\n";
                }
                for (const std::string& lexeme : slotted) {
                    std::size_t value = 0;
                    slot_count += ObservationDelexicalizer::is_slot(lexeme) ? 1 : 0;
                    literal_coordinates += coordinate_mentions(lexeme).size();
                    literal_numbers_ge2 += NumberWords::parse(lexeme, value) && value >= 2 ? 1 : 0;
                }
            }
        }
        const std::size_t field_total = (train.size() + eval.size()) * ObservationText::kFieldCount;
        require(round_trips == field_total, "delexicalize -> relexicalize restores every ground-truth field exactly");
        require(slot_count > 0, "ground truth contains slot-bound quantities");
        std::cout << "  " << round_trips << "/" << field_total << " fields round-trip; " << slot_count
                  << " slot tokens; literal coordinates left " << literal_coordinates << "; literal numbers >= 2 left "
                  << literal_numbers_ge2 << "\n";

        const TrainingSample& t1 = sample("syn_math_train_001");
        std::cout << "  train_001.pathway_activity slots: "
                  << LexemeSplitter::join(DualObjectiveDistillationBridge::target_lexemes(t1, 0)) << "\n";

        const std::vector<std::string> generated = LexemeSplitter::split(
            "<COUNT:DEAD_L1|W> of the <DIM:W_1|w> units ( units <LIST:DEAD_L1> ) never fire , and the intensity rises "
            "with <COORD:RISE_1> and <COORD:RISE_2> .");
        const TrainingSample& e4 = sample("syn_math_eval_004");
        std::string e4_text;
        require(ObservationDelexicalizer::relexicalize(generated, ShapeSlotLexicon::from(e4.measured), e4_text),
                "slot text fills on eval_004");
        std::cout << "  filled on eval_004 (" << e4.telemetry.shape().to_string() << "): " << e4_text << "\n";
        require(ObservationConsistency::verify_field(e4_text, e4.measured), "filled slot text is consistent with eval_004");

        const TrainingSample& t5 = sample("syn_math_train_005");
        std::string t5_text;
        const bool t5_ok = ObservationDelexicalizer::relexicalize(generated, ShapeSlotLexicon::from(t5.measured), t5_text);
        std::cout << "  same slots on train_005 (" << t5.telemetry.shape().to_string() << "): "
                  << (t5_ok ? t5_text : std::string("unfillable (no such quantity)")) << "\n";
        require(!t5_ok, "slot naming an absent quantity (no dead units on train_005) is reported unfillable");

        std::size_t literal_out_of_range = 0;
        std::size_t slot_out_of_range = 0;
        std::size_t filled_pairs = 0;
        for (const TrainingSample& source : train) {
            const ShapeSlotLexicon source_lex = ShapeSlotLexicon::from(source.measured);
            for (const TrainingSample* target : all_samples()) {
                const ShapeSlotLexicon target_lex = ShapeSlotLexicon::from(target->measured);
                const std::size_t d_in = target->telemetry.shape().input_dim;
                for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
                    const std::string& text = source.ground_truth_targets.observations.field(f);
                    for (std::size_t c : coordinate_mentions(text)) {
                        literal_out_of_range += c >= d_in ? 1 : 0;
                    }
                    std::vector<std::string> slotted = ObservationDelexicalizer::delexicalize(text, source_lex);
                    std::vector<std::string> slot_only;
                    for (const std::string& lexeme : slotted) {
                        slot_only.push_back(ObservationDelexicalizer::is_slot(lexeme) ? lexeme : "w");
                    }
                    std::string filled;
                    if (ObservationDelexicalizer::relexicalize(slot_only, target_lex, filled)) {
                        ++filled_pairs;
                        for (std::size_t c : coordinate_mentions(filled)) {
                            slot_out_of_range += c >= d_in ? 1 : 0;
                        }
                    }
                }
            }
        }
        std::cout << "  training text copied verbatim onto every network: " << literal_out_of_range
                  << " out-of-range coordinate mentions; slot-filled (" << filled_pairs << " fillable field pairs): "
                  << slot_out_of_range << "\n";
        require(literal_out_of_range > 0 && slot_out_of_range == 0, "slot filling never names a coordinate the network lacks");
    }

    void verify_decoder() {
        std::cout << "decoder (improvements 2, 3, 4)\n";
        DistillerConfig small;
        small.d_model = 16;
        small.decoder_hidden_dim = 12;
        small.num_latents = 4;
        small.max_field_tokens = 24;
        small.decoder_dropout = 0.0;
        small.decoder_output_weight_decay = 0.0;
        const std::vector<std::string> lexemes = LexemeSplitter::split("<COUNT:DEAD_L1|W> of the <DIM:W_1|w> units never fire .");
        DeterministicRng rng(77);
        Eigen::MatrixXd z(4, 16);
        for (Eigen::Index r = 0; r < z.rows(); ++r) {
            for (Eigen::Index c = 0; c < z.cols(); ++c) {
                z(r, c) = rng.normal(0.0, 1.0);
            }
        }
        Eigen::VectorXd d_hat(static_cast<Eigen::Index>(ObservationDescriptor::dim()));
        for (Eigen::Index i = 0; i < d_hat.size(); ++i) {
            d_hat(i) = rng.uniform();
        }

        AutoregressiveCausalDecoder decoder(small, 5);
        decoder.prepare_vocabulary({lexemes});
        const Eigen::MatrixXd alpha = decoder.attention_weights(z, d_hat, 0, lexemes);
        require(alpha.rows() == static_cast<Eigen::Index>(lexemes.size() + 1) && alpha.cols() == 4,
                "attention is T x M (one weight per latent per step)");
        require((alpha.rowwise().sum().array() - 1.0).abs().maxCoeff() < 1e-12, "attention rows sum to 1");
        require((alpha.rowwise().maxCoeff() - alpha.rowwise().minCoeff()).maxCoeff() > 1e-6,
                "attention weights differ across latents (not a mean)");

        Eigen::MatrixXd grad;
        const double loss = decoder.loss_and_latent_gradient(z, d_hat, 1, lexemes, grad);
        double worst = 0.0;
        for (Eigen::Index r = 0; r < z.rows(); ++r) {
            for (Eigen::Index c = 0; c < z.cols(); c += 3) {
                Eigen::MatrixXd plus = z;
                Eigen::MatrixXd minus = z;
                plus(r, c) += 1e-5;
                minus(r, c) -= 1e-5;
                const double numeric =
                    (decoder.compute_loss(plus, d_hat, 1, lexemes) - decoder.compute_loss(minus, d_hat, 1, lexemes)) / 2e-5;
                worst = std::max(worst, std::abs(numeric - grad(r, c)) / std::max(1e-6, std::abs(numeric) + std::abs(grad(r, c))));
            }
        }
        std::cout << "  loss " << loss << ", dLoss/dZ finite-difference worst relative error " << worst << "\n";
        require(worst < 1e-4, "cross-attention backprop into Z matches finite differences");
        double row_spread = 0.0;
        for (Eigen::Index r = 1; r < grad.rows(); ++r) {
            row_spread = std::max(row_spread, (grad.row(r) - grad.row(0)).norm());
        }
        std::cout << "  max |dZ_r - dZ_0| over latent rows " << row_spread << " (mean pooling gives 0)\n";
        require(row_spread > 1e-6, "each latent row receives its own gradient");

        Eigen::VectorXd other = d_hat;
        other(0) += 1.0;
        other(5) -= 0.7;
        const double with_dhat = decoder.compute_loss(z, d_hat, 2, lexemes);
        const double with_other = decoder.compute_loss(z, other, 2, lexemes);
        std::cout << "  loss with d_hat " << with_dhat << ", with altered d_hat " << with_other << "\n";
        require(std::abs(with_dhat - with_other) > 1e-9, "predicted descriptor conditions the decoder (W_aux d_hat)");
        require_throws([&]() { decoder.compute_loss(z, Eigen::VectorXd::Zero(3), 0, lexemes); },
                       "descriptor of the wrong size rejected");

        DistillerConfig dropped = small;
        dropped.decoder_dropout = 0.15;
        AutoregressiveCausalDecoder with_dropout(dropped, 5);
        with_dropout.prepare_vocabulary({lexemes});
        with_dropout.set_frozen(false);
        const double eval_loss = with_dropout.compute_loss(z, d_hat, 0, lexemes);
        Eigen::MatrixXd sink;
        const double train_loss = with_dropout.compute_loss_and_step(z, d_hat, 0, lexemes, sink);
        AutoregressiveCausalDecoder without_dropout(small, 5);
        without_dropout.prepare_vocabulary({lexemes});
        without_dropout.set_frozen(false);
        const double plain_eval = without_dropout.compute_loss(z, d_hat, 0, lexemes);
        const double plain_train = without_dropout.compute_loss_and_step(z, d_hat, 0, lexemes, sink);
        std::cout << "  dropout 0.15: eval-mode loss " << eval_loss << ", training-pass loss " << train_loss
                  << "; dropout 0: " << plain_eval << " / " << plain_train << "\n";
        require(std::abs(eval_loss - train_loss) > 1e-9, "dropout active in training steps");
        require(std::abs(plain_eval - plain_train) < 1e-12, "no dropout in eval mode");

        DistillerConfig decayed = small;
        decayed.decoder_output_weight_decay = 0.1;
        AutoregressiveCausalDecoder a(decayed, 9);
        AutoregressiveCausalDecoder b(small, 9);
        a.prepare_vocabulary({lexemes});
        b.prepare_vocabulary({lexemes});
        a.set_frozen(false);
        b.set_frozen(false);
        a.compute_loss_and_step(z, d_hat, 0, lexemes, sink);
        b.compute_loss_and_step(z, d_hat, 0, lexemes, sink);
        const double shrink = 1.0 - decayed.decoder_learning_rate * 0.1;
        const double decay_error = (a.output_weights() - shrink * b.output_weights()).cwiseAbs().maxCoeff();
        std::cout << "  ||W_out||: decay 0.1 " << a.output_weights().norm() << ", decay 0 " << b.output_weights().norm()
                  << ", max |W_decay - (1 - lr*0.1) W_plain| " << decay_error << "\n";
        require(decay_error < 1e-12 && a.output_weights().norm() < b.output_weights().norm(),
                "decoupled weight decay shrinks the vocabulary projection");
    }

    static Eigen::MatrixXd random_matrix(DeterministicRng& rng, Eigen::Index rows, Eigen::Index cols) {
        Eigen::MatrixXd m(rows, cols);
        for (Eigen::Index r = 0; r < rows; ++r) {
            for (Eigen::Index c = 0; c < cols; ++c) {
                m(r, c) = rng.normal(0.0, 1.0);
            }
        }
        return m;
    }

    static Eigen::VectorXd random_descriptor(DeterministicRng& rng) {
        Eigen::VectorXd d(static_cast<Eigen::Index>(ObservationDescriptor::dim()));
        for (Eigen::Index i = 0; i < d.size(); ++i) {
            d(i) = rng.uniform();
        }
        return d;
    }

    void verify_device_decoder() {
        std::cout << "decoder on the OpenCL device (CLBlast GEMM + OpenCL C kernels)\n";
        DistillerConfig small;
        small.d_model = 32;
        small.decoder_hidden_dim = 24;
        small.num_latents = 4;
        small.max_field_tokens = 40;
        DistillerConfig small_cpu = small;
        small_cpu.use_gpu = false;
        AutoregressiveCausalDecoder gpu(small, 11);
        AutoregressiveCausalDecoder cpu(small_cpu, 11);
        const std::string accelerator = gpu.accelerator();
        std::cout << "  accelerator: " << accelerator << "\n";
        require(cpu.accelerator() == "cpu", "use_gpu = false keeps the decoder on the CPU");
        if (accelerator == "cpu") {
            std::cout << "  no OpenCL GPU with double precision; device checks skipped\n";
            return;
        }

        const std::vector<std::vector<std::string>> sentences = {
            LexemeSplitter::split("<COUNT:DEAD_L1|W> of the <DIM:W_1|w> first-layer units never fire on any sample ."),
            LexemeSplitter::split("The output tracks x3 closely , with a correlation of 0.97 across samples ."),
            LexemeSplitter::split("Hidden activity spans two directions ; the leading one carries most of the variance ."),
        };
        const std::vector<std::string> unseen = LexemeSplitter::split("A gate closes whenever x5 drops below zero .");
        gpu.prepare_vocabulary(sentences);
        cpu.prepare_vocabulary(sentences);
        gpu.set_frozen(false);
        cpu.set_frozen(false);
        DeterministicRng rng(404);
        double worst_loss = 0.0;
        double worst_dz = 0.0;
        for (std::size_t step = 0; step < 9; ++step) {
            const Eigen::MatrixXd z = random_matrix(rng, 4, 32);
            const Eigen::VectorXd d_hat = random_descriptor(rng);
            const std::vector<std::string>& lexemes = step == 6 ? unseen : sentences[step % sentences.size()];
            Eigen::MatrixXd dz_gpu = Eigen::MatrixXd::Zero(4, 32);
            Eigen::MatrixXd dz_cpu = Eigen::MatrixXd::Zero(4, 32);
            const double loss_gpu = gpu.compute_loss_and_step(z, d_hat, step % 4, lexemes, dz_gpu);
            const double loss_cpu = cpu.compute_loss_and_step(z, d_hat, step % 4, lexemes, dz_cpu);
            worst_loss = std::max(worst_loss, std::abs(loss_gpu - loss_cpu));
            worst_dz = std::max(worst_dz, (dz_gpu - dz_cpu).cwiseAbs().maxCoeff());
        }
        std::cout << "  9 training steps with dropout 0.15 and decay 0.1 (step 7 adds new tokens): max |loss gpu - cpu| "
                  << worst_loss << ", max |dZ gpu - cpu| " << worst_dz << "\n";
        require(worst_loss < 1e-9, "device training-step losses match the CPU");
        require(worst_dz < 1e-9, "device dLoss/dZ matches the CPU");
        require(gpu.tokenizer().vocab_size() == cpu.tokenizer().vocab_size(), "vocabulary grows identically");

        const Eigen::MatrixXd z = random_matrix(rng, 4, 32);
        const Eigen::VectorXd d_hat = random_descriptor(rng);
        const double eval_gpu = gpu.compute_loss(z, d_hat, 1, sentences[1]);
        const double eval_cpu = cpu.compute_loss(z, d_hat, 1, sentences[1]);
        std::cout << "  eval-mode loss gpu " << eval_gpu << ", cpu " << eval_cpu << "\n";
        require(std::abs(eval_gpu - eval_cpu) < 1e-9, "device eval-mode loss matches the CPU");

        AutoregressiveCausalDecoder after_step(small, 11);
        after_step.prepare_vocabulary(sentences);
        after_step.set_frozen(false);
        Eigen::MatrixXd sink = Eigen::MatrixXd::Zero(4, 32);
        after_step.compute_loss_and_step(z, d_hat, 0, sentences[0], sink);
        require_throws([&]() { AutoregressiveCausalDecoder copy(after_step); },
                       "copying while the device holds newer parameters is rejected");

        const double weight_gap = (gpu.output_weights() - cpu.output_weights()).cwiseAbs().maxCoeff();
        const Eigen::MatrixXd attention_gap =
            gpu.attention_weights(z, d_hat, 2, sentences[2]) - cpu.attention_weights(z, d_hat, 2, sentences[2]);
        const std::vector<int> generated_gpu = gpu.generate_field_tokens(z, d_hat, 0, 40);
        const std::vector<int> generated_cpu = cpu.generate_field_tokens(z, d_hat, 0, 40);
        std::cout << "  after synchronizing: max |W_out gpu - cpu| " << weight_gap << ", max |attention gpu - cpu| "
                  << attention_gap.cwiseAbs().maxCoeff() << ", generated " << generated_gpu.size() << " / "
                  << generated_cpu.size() << " tokens\n";
        require(weight_gap < 1e-9, "synchronized device parameters match the CPU");
        require(attention_gap.cwiseAbs().maxCoeff() < 1e-9 && generated_gpu == generated_cpu,
                "generation from synchronized parameters matches the CPU");
        AutoregressiveCausalDecoder copy(gpu);
        require(std::abs(copy.compute_loss(z, d_hat, 1, sentences[1]) - eval_cpu) < 1e-9,
                "a copy made after synchronizing carries the trained parameters");

        DistillerConfig full;
        DistillerConfig full_cpu = full;
        full_cpu.use_gpu = false;
        AutoregressiveCausalDecoder full_gpu(full, 3);
        AutoregressiveCausalDecoder full_host(full_cpu, 3);
        std::vector<std::vector<std::string>> fields;
        for (std::size_t s = 0; s < 8; ++s) {
            std::vector<std::string> lexemes;
            for (std::size_t i = 0; i < 60; ++i) {
                lexemes.push_back("w" + std::to_string((s * 211 + i * 29) % 1700));
            }
            fields.push_back(lexemes);
        }
        full_gpu.prepare_vocabulary(fields);
        full_host.prepare_vocabulary(fields);
        full_gpu.set_frozen(false);
        full_host.set_frozen(false);
        const Eigen::MatrixXd z_full = random_matrix(rng, 8, 256);
        const Eigen::VectorXd d_full = random_descriptor(rng);
        Eigen::MatrixXd dz_full = Eigen::MatrixXd::Zero(8, 256);
        const auto time_steps = [&](AutoregressiveCausalDecoder& decoder) {
            decoder.compute_loss_and_step(z_full, d_full, 0, fields[0], dz_full);
            const auto started = std::chrono::steady_clock::now();
            for (std::size_t step = 1; step <= 16; ++step) {
                decoder.compute_loss_and_step(z_full, d_full, step % 4, fields[step % fields.size()], dz_full);
            }
            decoder.synchronize_host();
            return std::chrono::duration<double>(std::chrono::steady_clock::now() - started).count() / 16.0;
        };
        const double cpu_seconds = time_steps(full_host);
        const double gpu_seconds = time_steps(full_gpu);
        std::cout << "  d_model 256, hidden 256, 8 latents, vocab " << full_gpu.tokenizer().vocab_size()
                  << ", 61-token fields: training step cpu " << cpu_seconds * 1000.0 << " ms, gpu " << gpu_seconds * 1000.0
                  << " ms (" << cpu_seconds / gpu_seconds << "x)\n";
        require(gpu_seconds < cpu_seconds, "device training step is faster than the CPU at full size");
    }

    void verify_training() {
        std::cout << "training\n";
        const std::vector<TrainingSample> subset(train.begin(), train.begin() + std::min<std::size_t>(12, train.size()));
        DualObjectiveDistillationBridge bridge(config);
        bridge.prepare_vocabulary(subset);
        const std::vector<LossBreakdown> pre = bridge.run_pretraining_phase(subset, 20, 0.05);
        require(pre.back().aux1_observation_loss < pre.front().aux1_observation_loss, "step 1 lowers Aux1");
        require(pre.back().aux2_eigen_sign_loss < pre.front().aux2_eigen_sign_loss, "step 1 lowers Aux2");
        require(pre.back().token_cross_entropy == 0.0, "decoder frozen in step 1");

        const double ce_before = bridge.evaluate_sample(subset.front()).token_cross_entropy;
        const FinetuneOutcome memorize = bridge.run_finetuning_phase(subset, 40, 0.05, subset);
        const double ce_after = bridge.evaluate_sample(subset.front()).token_cross_entropy;
        const LossBreakdown first_epoch = memorize.history.front().train;
        const LossBreakdown last_epoch = memorize.history.back().train;
        require(last_epoch.token_cross_entropy < 0.5 * first_epoch.token_cross_entropy, "step 2 lowers token CE");
        require(ce_after < ce_before, "evaluate_sample CE falls");
        std::cout << "  token CE " << first_epoch.token_cross_entropy << " -> " << last_epoch.token_cross_entropy << " over "
                  << memorize.history.size() << " epochs\n";

        double train_f1 = 0.0;
        std::size_t filled_fields = 0;
        for (const TrainingSample& s : subset) {
            const DistillationResult r = bridge.distill_telemetry(s.telemetry);
            train_f1 += ObservationSimilarity::mean_f1(r.decoder_observations, s.ground_truth_targets.observations);
            for (bool ok : r.decoder_slots_filled) {
                filled_fields += ok ? 1 : 0;
            }
        }
        train_f1 /= static_cast<double>(subset.size());
        std::cout << "  decoder word F1 on training records " << train_f1 << ", slot-filled fields " << filled_fields << "/"
                  << subset.size() * ObservationText::kFieldCount << "\n";
        require(train_f1 > 0.6, "decoder reproduces training observations");

        const FinetuneOutcome outcome = bridge.run_finetuning_phase(subset, 30, 0.05, eval);
        double lowest_total = 1e300;
        std::size_t lowest_epoch = 0;
        std::cout << "  held-out token CE per epoch:";
        for (const EpochRecord& r : outcome.history) {
            std::cout << " " << r.eval.token_cross_entropy << (r.best_so_far ? "*" : "");
            if (r.eval.total_loss < lowest_total) {
                lowest_total = r.eval.total_loss;
                lowest_epoch = r.epoch;
            }
        }
        std::cout << "\n  ran " << outcome.history.size() << "/30, stopped_early " << outcome.stopped_early << ", best epoch "
                  << outcome.best_epoch << "\n";
        bool trailing_without_decrease = true;
        if (outcome.stopped_early) {
            double best_before = 1e300;
            const std::size_t n = outcome.history.size();
            const std::size_t p = config.early_stopping_patience;
            for (std::size_t i = 0; i + p < n; ++i) {
                best_before = std::min(best_before, outcome.history[i].eval.token_cross_entropy);
            }
            for (std::size_t i = n - p; i < n; ++i) {
                trailing_without_decrease = trailing_without_decrease && outcome.history[i].eval.token_cross_entropy >= best_before;
            }
        }
        require(outcome.stopped_early ? trailing_without_decrease && outcome.history.size() < 30 : outcome.history.size() == 30,
                "early stopping fires only after 5 epochs without a held-out CE decrease");
        require(outcome.best_epoch == lowest_epoch, "best epoch is the lowest held-out total");
        const double restored_total = bridge.evaluate_dataset(eval).total_loss;
        std::cout << "  held-out total after restore " << restored_total << ", recorded at best epoch " << lowest_total << "\n";
        require(std::abs(restored_total - lowest_total) < 1e-9, "best-epoch weights restored");

        const std::filesystem::path artifact_root = std::filesystem::temp_directory_path() / "forward_analysis_test_artifacts";
        std::filesystem::remove_all(artifact_root);
        TrainingRunArtifactWriter artifacts_probe(artifact_root, "run_test");
        artifacts_probe.record_epoch("finetune", 1, outcome.history.back().train, outcome.history.back().eval);
        for (const TrainingSample& s : eval) {
            const DistillationResult r = bridge.distill_telemetry(s.telemetry);
            require(r.channel_count == s.telemetry.shape().channel_count(), s.sample_id + " channel count reported");
            require(r.predicted_eigen_signs.size() == 4 &&
                        r.predicted_observation_descriptor.size() == static_cast<Eigen::Index>(ObservationDescriptor::dim()),
                    s.sample_id + " aux predictions sized");
            for (std::size_t f = 0; f < ObservationText::kFieldCount; ++f) {
                require(ObservationConsistency::verify_field(r.observations.field(f), r.measured),
                        s.sample_id + " final " + ObservationText::field_names()[f] + " consistent");
                require(r.decoder_token_counts[f] >= 2 && r.decoder_token_counts[f] <= config.max_field_tokens,
                        s.sample_id + " decoder budget respected");
            }
            artifacts_probe.record_sample("eval", s, r);
            std::cout << "  " << s.sample_id << " word F1 decoder "
                      << ObservationSimilarity::mean_f1(r.decoder_observations, s.ground_truth_targets.observations)
                      << ", output " << ObservationSimilarity::mean_f1(r.observations, s.ground_truth_targets.observations)
                      << "\n";
        }

        TrainingRunSettings settings;
        settings.finetune_epochs = 40;
        artifacts_probe.finish(settings, config, train, eval);
        const std::filesystem::path dir = artifacts_probe.directory();
        for (const char* name : {"run.json", "loss_history.csv", "samples.csv", "fields.csv", "samples/syn_math_eval_001.json"}) {
            require(std::filesystem::exists(dir / name), std::string("artifact ") + name + " written");
        }
        std::ifstream run_file(dir / "run.json");
        const nlohmann::json run = nlohmann::json::parse(run_file);
        require(run.at("metrics").at("eval").at("samples") == eval.size() &&
                    run.at("metrics").at("eval").at("total_fields") == eval.size() * ObservationText::kFieldCount,
                "run.json aggregates the eval split");
        std::ifstream fields_file(dir / "fields.csv");
        std::size_t field_lines = 0;
        for (std::string line; std::getline(fields_file, line);) {
            ++field_lines;
        }
        require(field_lines == 1 + eval.size() * ObservationText::kFieldCount,
                "fields.csv has a header and one row per record x field");
        run_file.close();
        fields_file.close();
        std::filesystem::remove_all(artifact_root);
    }
};

int main() {
    std::cout << std::unitbuf;
    ForwardAnalysisTestSuite suite;
    try {
        suite.verify_math();
        suite.verify_text_helpers();
        suite.verify_builder();
        suite.verify_importer();
        suite.verify_extractor();
        suite.verify_corrected_dataset();
        suite.verify_encoder();
        suite.verify_domain_feature_extractors();
        suite.verify_multi_family_tokens();
        suite.verify_domain_prediction_heads();
        suite.verify_description();
        suite.verify_slots();
        suite.verify_decoder();
        suite.verify_device_decoder();
        suite.verify_training();
    } catch (const std::exception& e) {
        std::cout << "unexpected exception: " << e.what() << "\n";
        ++suite.failed;
    }
    std::cout << suite.passed << " passed, " << suite.failed << " failed\n";
    return suite.failed == 0 ? 0 : 1;
}
