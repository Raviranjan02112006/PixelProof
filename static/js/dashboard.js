
/* ============================================================
   PIXELPROOF — DASHBOARD.JS
   Dashboard Rendering and UI Management
   ============================================================ */

const Dashboard = (() => {

    /* ========================================================
       1. GENERAL HELPERS
       ======================================================== */

    function element(id) {
        return document.getElementById(id);
    }


    function setText(id, value) {

        const target = element(id);

        if (!target) return;

        target.textContent =
            value === undefined || value === null
                ? "N/A"
                : String(value);
    }


    function getFirstValue(object, keys) {

        if (!object || typeof object !== "object") {
            return undefined;
        }

        for (const key of keys) {

            if (
                object[key] !== undefined &&
                object[key] !== null
            ) {
                return object[key];
            }
        }

        return undefined;
    }


    function toNumber(value) {

        if (
            value === undefined ||
            value === null ||
            value === ""
        ) {
            return NaN;
        }

        const number = Number(value);

        return Number.isFinite(number)
            ? number
            : NaN;
    }


    function formatEvidence(value, fieldName = "") {

        const number = toNumber(value);

        if (!Number.isFinite(number)) {
            return "N/A";
        }

        const isPercentageField =
            String(fieldName).toLowerCase().includes("percentage") ||
            String(fieldName).toLowerCase().includes("percent");

        const percentageValue =
            !isPercentageField &&
            number >= 0 &&
            number <= 1
                ? number * 100
                : number;

        return percentageValue.toFixed(2) + "%";
    }


    function formatThreshold(value) {

        const number = toNumber(value);

        if (!Number.isFinite(number)) {
            return "N/A";
        }

        const percentageValue =
            number >= 0 && number <= 1
                ? number * 100
                : number;

        return percentageValue.toFixed(2) + "%";
    }


    function formatWeight(value) {

        const number = toNumber(value);

        if (!Number.isFinite(number)) {
            return "N/A";
        }

        const percentageValue =
            number >= 0 && number <= 1
                ? number * 100
                : number;

        return percentageValue.toFixed(2) + "%";
    }


    function setPredictionColor(id, prediction) {

        const target = element(id);

        if (!target) return;

        target.classList.remove(
            "text-real",
            "text-fake",
            "text-pending"
        );

        const value = String(
            prediction || ""
        ).toUpperCase();

        if (value.includes("REAL")) {

            target.classList.add("text-real");

        } else if (value.includes("FAKE")) {

            target.classList.add("text-fake");

        } else {

            target.classList.add("text-pending");
        }
    }


    function createCell(value) {

        const cell = document.createElement("td");

        cell.textContent =
            value === undefined || value === null
                ? "N/A"
                : String(value);

        return cell;
    }


    function createEmptyTableMessage(message) {

        const tableBody = element("individualModelResults");

        if (!tableBody) return;

        const row = document.createElement("tr");
        const cell = document.createElement("td");

        cell.colSpan = 5;
        cell.textContent = message;

        row.appendChild(cell);
        tableBody.appendChild(row);
    }


    /* ========================================================
       2. INDIVIDUAL MODEL TABLE
       ======================================================== */

    function renderIndividualModels(models) {

        const tableBody = element("individualModelResults");

        if (!tableBody) return;

        tableBody.replaceChildren();

        if (!models) {

            createEmptyTableMessage(
                "Individual model results are not available."
            );

            setText(
                "individualModelSummary",
                "Individual model results are not available."
            );

            return;
        }


        let modelList = [];


        if (Array.isArray(models)) {

            modelList = models;

        } else if (typeof models === "object") {

            if (Array.isArray(models.models)) {

                modelList = models.models;

            } else {

                modelList = Object.entries(models).map(
                    ([name, result]) => ({
                        name,
                        ...(result || {})
                    })
                );
            }
        }


        if (modelList.length === 0) {

            createEmptyTableMessage(
                "No individual model results were returned."
            );

            setText(
                "individualModelSummary",
                "No individual model results were returned."
            );

            return;
        }


        modelList.forEach((model, index) => {

            const row = document.createElement("tr");

            const name = getFirstValue(
                model,
                [
                    "name",
                    "model_name",
                    "model",
                    "display_name"
                ]
            ) || `Model ${index + 1}`;


            const prediction = getFirstValue(
                model,
                [
                    "prediction",
                    "predicted_class",
                    "final_prediction",
                    "label"
                ]
            );


            const fakeField = [
                "fake_evidence_percentage",
                "fake_percentage",
                "fake_evidence",
                "fake_score"
            ].find(key =>
                model[key] !== undefined &&
                model[key] !== null
            );


            const realField = [
                "real_evidence_percentage",
                "real_percentage",
                "real_evidence",
                "real_score"
            ].find(key =>
                model[key] !== undefined &&
                model[key] !== null
            );


            const fakeValue = fakeField
                ? model[fakeField]
                : undefined;

            let realValue = realField
                ? model[realField]
                : undefined;


            if (
                realValue === undefined &&
                fakeValue !== undefined
            ) {

                const fakeNumber = toNumber(fakeValue);

                if (Number.isFinite(fakeNumber)) {

                    const fakePercentage =
                        fakeField.includes("percentage") ||
                        fakeField.includes("percent")
                            ? fakeNumber
                            : fakeNumber >= 0 && fakeNumber <= 1
                                ? fakeNumber * 100
                                : fakeNumber;

                    realValue = 100 - fakePercentage;
                }
            }


            const weight = getFirstValue(
                model,
                [
                    "weight_percentage",
                    "weight",
                    "model_weight",
                    "normalized_weight"
                ]
            );


            row.appendChild(createCell(name));

            row.appendChild(
                createCell(
                    formatEvidence(fakeValue, fakeField)
                )
            );

            row.appendChild(
                createCell(
                    formatEvidence(realValue, realField)
                )
            );

            row.appendChild(
                createCell(formatWeight(weight))
            );


            const predictionCell = createCell(prediction);

            const predictionText = String(
                prediction || ""
            ).toUpperCase();

            if (predictionText.includes("REAL")) {

                predictionCell.classList.add("text-real");

            } else if (predictionText.includes("FAKE")) {

                predictionCell.classList.add("text-fake");

            } else {

                predictionCell.classList.add("text-pending");
            }

            row.appendChild(predictionCell);

            tableBody.appendChild(row);
        });


        setText(
            "individualModelSummary",
            `${modelList.length} individual model results displayed.`
        );
    }


    /* ========================================================
       3. MATHEMATICAL ENSEMBLE
       ======================================================== */

    function renderMathematicalResult(result) {

        if (!result) return;


        const prediction = getFirstValue(
            result,
            [
                "prediction",
                "final_prediction",
                "predicted_class"
            ]
        );


        const realField = [
            "real_support_percentage",
            "real_evidence_percentage",
            "real_support",
            "real_score"
        ].find(key =>
            result[key] !== undefined &&
            result[key] !== null
        );


        const fakeField = [
            "fake_support_percentage",
            "fake_evidence_percentage",
            "fake_support",
            "fake_score"
        ].find(key =>
            result[key] !== undefined &&
            result[key] !== null
        );


        const real = realField
            ? result[realField]
            : undefined;

        const fake = fakeField
            ? result[fakeField]
            : undefined;


        const threshold = getFirstValue(
            result,
            [
                "ensemble_threshold",
                "threshold"
            ]
        );


        setText("mathPrediction", prediction);

        setText(
            "mathRealSupport",
            formatEvidence(real, realField)
        );

        setText(
            "mathFakeSupport",
            formatEvidence(fake, fakeField)
        );

        setText(
            "mathThreshold",
            formatThreshold(threshold)
        );

        setPredictionColor(
            "mathPrediction",
            prediction
        );
    }


    /* ========================================================
       4. CNN RESULT
       ======================================================== */

    function renderCNNResult(result) {

        if (!result) return;


        const prediction = getFirstValue(
            result,
            [
                "prediction",
                "final_prediction",
                "predicted_class"
            ]
        );


        const realField = [
            "real_evidence_percentage",
            "real_percentage",
            "real_evidence",
            "real_score"
        ].find(key =>
            result[key] !== undefined &&
            result[key] !== null
        );


        const fakeField = [
            "fake_evidence_percentage",
            "fake_percentage",
            "fake_evidence",
            "fake_score"
        ].find(key =>
            result[key] !== undefined &&
            result[key] !== null
        );


        const real = realField
            ? result[realField]
            : undefined;

        const fake = fakeField
            ? result[fakeField]
            : undefined;


        const threshold = getFirstValue(
            result,
            [
                "threshold",
                "classification_threshold"
            ]
        );


        setText("cnnPrediction", prediction);

        setText(
            "cnnRealEvidence",
            formatEvidence(real, realField)
        );

        setText(
            "cnnFakeEvidence",
            formatEvidence(fake, fakeField)
        );

        setText(
            "cnnThreshold",
            formatThreshold(threshold)
        );


        setText(
            "cnnDetails",
            result.note ||
            result.description ||
            "ResNet50 prediction received."
        );


        setPredictionColor(
            "cnnPrediction",
            prediction
        );


        setText(
            "cnnAnalysisStatus",
            "CNN analysis completed."
        );

        setText(
            "cnnAnalysisPrediction",
            prediction
        );

        setPredictionColor(
            "cnnAnalysisPrediction",
            prediction
        );
    }


    /* ========================================================
       5. HYBRID RESULT
       ======================================================== */

    function renderHybridResult(result) {

        if (!result) return;


        /*
           Backend hybrid_prediction structure:

           {
               final_prediction: "FAKE",

               hybrid_real_evidence_percentage: 23.04,

               hybrid_fake_evidence_percentage: 76.96,

               model_agreement: {
                   status: "AGREEMENT",
                   mathematical_prediction: "FAKE",
                   cnn_prediction: "FAKE",
                   models_agree: true
               }
           }
        */


        const prediction = getFirstValue(
            result,
            [
                "final_prediction",
                "prediction",
                "predicted_class"
            ]
        );


        /*
           IMPORTANT:
           Read the exact hybrid evidence field names
           returned by hybrid_prediction.py.
        */

        const realField = [
            "hybrid_real_evidence_percentage",
            "real_evidence_percentage",
            "real_support_percentage",
            "real_score"
        ].find(key =>
            result[key] !== undefined &&
            result[key] !== null
        );


        const fakeField = [
            "hybrid_fake_evidence_percentage",
            "fake_evidence_percentage",
            "fake_support_percentage",
            "fake_score"
        ].find(key =>
            result[key] !== undefined &&
            result[key] !== null
        );


        const real = realField
            ? result[realField]
            : undefined;

        const fake = fakeField
            ? result[fakeField]
            : undefined;


        /*
           Handle model_agreement as an object.
        */

        const agreementData = getFirstValue(
            result,
            [
                "model_agreement",
                "agreement"
            ]
        );


        let agreementText = "N/A";


        if (
            agreementData &&
            typeof agreementData === "object"
        ) {

            const status = getFirstValue(
                agreementData,
                [
                    "status",
                    "agreement_status"
                ]
            );

            const modelsAgree = getFirstValue(
                agreementData,
                [
                    "models_agree",
                    "agree"
                ]
            );


            if (status !== undefined) {

                agreementText = String(status)
                    .replaceAll("_", " ");

            } else if (typeof modelsAgree === "boolean") {

                agreementText = modelsAgree
                    ? "AGREEMENT"
                    : "DISAGREEMENT";
            }

        } else if (typeof agreementData === "boolean") {

            agreementText = agreementData
                ? "AGREEMENT"
                : "DISAGREEMENT";

        } else if (agreementData !== undefined) {

            agreementText = String(agreementData);
        }


        /*
           Update hybrid prediction.
        */

        setText(
            "hybridPrediction",
            prediction
        );


        setText(
            "hybridRealEvidence",
            formatEvidence(real, realField)
        );


        setText(
            "hybridFakeEvidence",
            formatEvidence(fake, fakeField)
        );


        setText(
            "modelAgreement",
            agreementText
        );


        setPredictionColor(
            "hybridPrediction",
            prediction
        );
    }


    /* ========================================================
       6. EVIDENCE BARS
       ======================================================== */

    function updateEvidenceBar(id, value) {

        const target = element(id);

        if (!target) return;

        let percentageValue = toNumber(value);

        if (!Number.isFinite(percentageValue)) {
            percentageValue = 0;
        }


        if (
            percentageValue >= 0 &&
            percentageValue <= 1
        ) {
            percentageValue *= 100;
        }


        percentageValue = Math.max(
            0,
            Math.min(100, percentageValue)
        );


        target.style.width =
            percentageValue + "%";
    }


    function renderEvidenceBars(data) {

        if (!data) return;


        const math = data.mathematical_prediction ||
            data.mathematical_result;

        const cnn = data.cnn_prediction ||
            data.cnn_result;

        const hybrid = data.hybrid_prediction ||
            data.hybrid_result;


        if (math) {

            updateEvidenceBar(
                "mathRealFill",
                math.real_support_percentage ??
                math.real_evidence_percentage ??
                math.real_support
            );

            updateEvidenceBar(
                "mathFakeFill",
                math.fake_support_percentage ??
                math.fake_evidence_percentage ??
                math.fake_support
            );
        }


        if (cnn) {

            updateEvidenceBar(
                "cnnRealFill",
                cnn.real_evidence_percentage ??
                cnn.real_percentage ??
                cnn.real_score
            );

            updateEvidenceBar(
                "cnnFakeFill",
                cnn.fake_evidence_percentage ??
                cnn.fake_percentage ??
                cnn.fake_score
            );
        }


        if (hybrid) {

            updateEvidenceBar(
                "hybridRealFill",
                hybrid.hybrid_real_evidence_percentage ??
                hybrid.real_evidence_percentage ??
                hybrid.real_support_percentage ??
                hybrid.real_score
            );

            updateEvidenceBar(
                "hybridFakeFill",
                hybrid.hybrid_fake_evidence_percentage ??
                hybrid.fake_evidence_percentage ??
                hybrid.fake_support_percentage ??
                hybrid.fake_score
            );
        }
    }


    /* ========================================================
       7. MAIN DASHBOARD RENDER FUNCTION
       ======================================================== */

    function render(data) {

        if (!data || typeof data !== "object") {

            console.error(
                "Invalid PixelProof dashboard data."
            );

            return;
        }


        renderIndividualModels(
            data.individual_models ||
            data.individual_model_results
        );


        renderMathematicalResult(
            data.mathematical_prediction ||
            data.mathematical_result
        );


        renderCNNResult(
            data.cnn_prediction ||
            data.cnn_result
        );


        renderHybridResult(
            data.hybrid_prediction ||
            data.hybrid_result
        );


        renderEvidenceBars(data);


        const dashboard = element(
            "predictionDashboard"
        );

        if (dashboard) {
            dashboard.style.display = "block";
        }
    }


    /* ========================================================
       8. RESET DASHBOARD
       ======================================================== */

    function reset() {

        const dashboard = element(
            "predictionDashboard"
        );

        if (dashboard) {
            dashboard.style.display = "none";
        }


        const tableBody = element(
            "individualModelResults"
        );

        if (tableBody) {

            tableBody.replaceChildren();

            createEmptyTableMessage(
                "Individual model results will appear after analysis."
            );
        }


        const ids = [

            "individualModelSummary",

            "mathPrediction",
            "mathRealSupport",
            "mathFakeSupport",
            "mathThreshold",

            "cnnPrediction",
            "cnnRealEvidence",
            "cnnFakeEvidence",
            "cnnThreshold",
            "cnnDetails",
            "cnnAnalysisStatus",
            "cnnAnalysisPrediction",

            "hybridPrediction",
            "hybridRealEvidence",
            "hybridFakeEvidence",
            "modelAgreement"
        ];


        ids.forEach(id => {
            setText(id, "Waiting for analysis");
        });


        setPredictionColor(
            "mathPrediction",
            ""
        );

        setPredictionColor(
            "cnnPrediction",
            ""
        );

        setPredictionColor(
            "cnnAnalysisPrediction",
            ""
        );

        setPredictionColor(
            "hybridPrediction",
            ""
        );


        [
            "mathRealFill",
            "mathFakeFill",
            "cnnRealFill",
            "cnnFakeFill",
            "hybridRealFill",
            "hybridFakeFill"

        ].forEach(id => {

            updateEvidenceBar(id, 0);
        });
    }


    /* ========================================================
       9. PUBLIC FUNCTIONS
       ======================================================== */

    return {

        render,

        reset,

        renderIndividualModels,

        renderMathematicalResult,

        renderCNNResult,

        renderHybridResult
    };

})();


/* ============================================================
   10. MAKE DASHBOARD AVAILABLE TO SCRIPT.JS
   ============================================================ */

window.PixelProofDashboard = Dashboard;

console.log(
    "PixelProof Dashboard JavaScript loaded successfully."
);
