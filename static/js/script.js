
/* ============================================================
   PIXELPROOF — SCRIPT.JS
   Image Upload + Mathematical Visualizations
   + Individual Models + Ensemble + CNN + Hybrid
   ============================================================ */


/* ============================================================
   1. HTML ELEMENTS
   ============================================================ */

const imageInput = document.getElementById("imageInput");
const dropArea = document.getElementById("dropArea");
const browseButton = document.getElementById("browseButton");
const analyseButton = document.getElementById("analyseButton");
const removeButton = document.getElementById("removeButton");

const uploadContent = document.getElementById("uploadContent");
const previewArea = document.getElementById("previewArea");
const imagePreview = document.getElementById("imagePreview");
const fileName = document.getElementById("fileName");

const statusMessage = document.getElementById("statusMessage");
const resultPanel = document.getElementById("resultPanel");

const analysisDashboard = document.getElementById("analysis");
const predictionDashboard = document.getElementById("predictionDashboard");

const resultText = document.getElementById("resultText");
const confidenceFill = document.getElementById("confidenceFill");
const confidenceText = document.getElementById("confidenceText");


/* ============================================================
   2. APPLICATION STATE
   ============================================================ */

let selectedFile = null;
let previewURL = null;
let analysisInProgress = false;


/* ============================================================
   3. GENERAL HELPER FUNCTIONS
   ============================================================ */

function getElement(id) {
    return document.getElementById(id);
}


function setText(id, value) {

    const target = getElement(id);

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


/*
   Converts evidence to a percentage.

   Explicit percentage fields are already percentages.
   Generic score fields between 0 and 1 are fractional scores.
*/

function getEvidencePercentage(result, keys) {

    if (!result) return NaN;

    for (const key of keys) {

        if (
            result[key] === undefined ||
            result[key] === null
        ) {
            continue;
        }

        const number = toNumber(result[key]);

        if (!Number.isFinite(number)) {
            continue;
        }

        const isPercentageField =
            key.toLowerCase().includes("percentage") ||
            key.toLowerCase().includes("percent");

        if (
            !isPercentageField &&
            number >= 0 &&
            number <= 1
        ) {
            return number * 100;
        }

        return number;
    }

    return NaN;
}


function formatPercentage(value) {

    const number = toNumber(value);

    if (!Number.isFinite(number)) {
        return "N/A";
    }

    return number.toFixed(2) + "%";
}


function addCacheBuster(url) {

    if (!url || typeof url !== "string") {
        return "";
    }

    const separator = url.includes("?") ? "&" : "?";

    return url + separator + "v=" + Date.now();
}


function setImage(id, url) {

    const target = getElement(id);

    if (!target || !url) {
        return;
    }

    target.onerror = () => {

        console.error(
            "Unable to load visualization:",
            url
        );

        target.style.display = "none";
    };

    target.onload = () => {
        target.style.display = "block";
    };

    target.src = addCacheBuster(url);
}


function setPredictionColor(id, prediction) {

    const target = getElement(id);

    if (!target) return;

    target.classList.remove(
        "prediction-real",
        "prediction-fake",
        "prediction-pending"
    );

    const value = String(
        prediction || ""
    ).toUpperCase();

    if (value.includes("REAL")) {

        target.classList.add("prediction-real");

    } else if (value.includes("FAKE")) {

        target.classList.add("prediction-fake");

    } else {

        target.classList.add("prediction-pending");
    }
}


function setStatus(message) {

    if (statusMessage) {
        statusMessage.textContent = message;
    }
}


function setButtonLoading(isLoading) {

    if (!analyseButton) return;

    if (isLoading) {

        analyseButton.disabled = true;
        analyseButton.textContent = "Analysing Image...";

    } else {

        analyseButton.disabled = !selectedFile;
        analyseButton.textContent = "✦ Analyse Image";
    }
}


/* ============================================================
   4. IMAGE UPLOAD
   ============================================================ */

// Open file selector.

if (browseButton && imageInput) {

    browseButton.addEventListener("click", () => {

        if (!analysisInProgress) {
            imageInput.click();
        }
    });
}


// Select image.

if (imageInput) {

    imageInput.addEventListener("change", (event) => {

        const file = event.target.files[0];

        if (file) {
            handleImage(file);
        }
    });
}


/* ============================================================
   5. IMAGE VALIDATION AND PREVIEW
   ============================================================ */

function handleImage(file) {

    if (analysisInProgress) {
        return;
    }


    if (!file) {
        return;
    }


    const allowedTypes = [
        "image/jpeg",
        "image/png"
    ];

    const allowedExtensions = [
        ".jpg",
        ".jpeg",
        ".png"
    ];

    const fileExtension = file.name
        .substring(file.name.lastIndexOf("."))
        .toLowerCase();


    const validType =
        allowedTypes.includes(file.type) ||
        (
            !file.type &&
            allowedExtensions.includes(fileExtension)
        );


    if (!validType) {

        setStatus(
            "Please select a JPG, JPEG or PNG image."
        );

        if (imageInput) {
            imageInput.value = "";
        }

        return;
    }


    if (file.size > 10 * 1024 * 1024) {

        setStatus(
            "Image size must be less than 10 MB."
        );

        if (imageInput) {
            imageInput.value = "";
        }

        return;
    }


    // Release previous preview URL.

    if (previewURL) {

        URL.revokeObjectURL(previewURL);

        previewURL = null;
    }


    selectedFile = file;

    previewURL = URL.createObjectURL(file);


    if (imagePreview) {
        imagePreview.src = previewURL;
    }

    if (fileName) {
        fileName.textContent = file.name;
    }


    if (uploadContent) {
        uploadContent.style.display = "none";
    }

    if (previewArea) {
        previewArea.style.display = "block";
    }


    // Reset previous analysis before a new image.

    resetResults();


    if (analyseButton) {
        analyseButton.disabled = false;
    }


    setStatus(
        "Image selected successfully. Ready for analysis."
    );
}


/* ============================================================
   6. DRAG AND DROP
   ============================================================ */

if (dropArea) {

    dropArea.addEventListener("dragover", (event) => {

        event.preventDefault();

        if (!analysisInProgress) {
            dropArea.classList.add("dragover");
        }
    });


    dropArea.addEventListener("dragleave", (event) => {

        event.preventDefault();

        dropArea.classList.remove("dragover");
    });


    dropArea.addEventListener("drop", (event) => {

        event.preventDefault();

        dropArea.classList.remove("dragover");

        if (analysisInProgress) {
            return;
        }

        const file = event.dataTransfer.files[0];

        if (file) {

            handleImage(file);

            if (imageInput) {
                imageInput.value = "";
            }
        }
    });
}


/* ============================================================
   7. RESET RESULTS
   ============================================================ */

function resetResults() {

    // Hide final result.

    if (resultPanel) {
        resultPanel.style.display = "none";
    }


    // Hide visualization dashboard.

    if (analysisDashboard) {
        analysisDashboard.style.display = "none";
    }


    // Reset prediction dashboard through dashboard.js.

    if (
        window.PixelProofDashboard &&
        typeof window.PixelProofDashboard.reset === "function"
    ) {

        window.PixelProofDashboard.reset();

    } else if (predictionDashboard) {

        predictionDashboard.style.display = "none";
    }


    // Clear old visualization images.

    [
        "fourierImage",
        "correlationImage",
        "spatialImage",
        "depthImage"
    ].forEach(id => {

        const target = getElement(id);

        if (target) {

            target.removeAttribute("src");
            target.style.display = "none";
        }
    });


    // Reset visualization descriptions.

    setText(
        "fourierDetails",
        "Fourier analysis results will appear here."
    );

    setText(
        "correlationDetails",
        "Correlation analysis results will appear here."
    );

    setText(
        "spatialDetails",
        "Spatial analysis results will appear here."
    );

    setText(
        "depthDetails",
        "Depth estimation results will appear here."
    );


    // Reset final result fields.

    setText(
        "resultText",
        "Your final prediction will appear here."
    );

    setText(
        "confidenceText",
        "Final hybrid evidence will appear here."
    );

    if (confidenceFill) {
        confidenceFill.style.width = "0%";
    }
}


/* ============================================================
   8. REMOVE SELECTED IMAGE
   ============================================================ */

if (removeButton) {

    removeButton.addEventListener("click", () => {

        if (analysisInProgress) {
            return;
        }


        selectedFile = null;


        if (previewURL) {

            URL.revokeObjectURL(previewURL);

            previewURL = null;
        }


        if (imageInput) {
            imageInput.value = "";
        }

        if (imagePreview) {
            imagePreview.removeAttribute("src");
        }


        if (previewArea) {
            previewArea.style.display = "none";
        }

        if (uploadContent) {
            uploadContent.style.display = "block";
        }


        resetResults();


        if (analyseButton) {
            analyseButton.disabled = true;
        }


        setStatus(
            "Select an image to begin."
        );
    });
}


/* ============================================================
   9. MATHEMATICAL VISUALIZATION RESULTS
   ============================================================ */

function formatModuleDetails(details, fallback) {

    if (
        details === undefined ||
        details === null
    ) {
        return fallback;
    }


    if (typeof details === "string") {
        return details;
    }


    if (typeof details !== "object") {
        return String(details);
    }


    const description = details.description || fallback;

    const additionalDetails = Object.entries(details)

        .filter(([key, value]) => {

            return (
                key !== "description" &&
                value !== undefined &&
                value !== null
            );
        })

        .map(([key, value]) => {

            const readableKey = key
                .replace(/_/g, " ")
                .replace(/\b\w/g, character =>
                    character.toUpperCase()
                );

            return `${readableKey}: ${value}`;
        });


    return [
        description,
        ...additionalDetails
    ].join(" | ");
}


function showAnalysis(data) {

    if (!data || typeof data !== "object") {
        return;
    }


    /*
       Current Flask response:

       data.mathematical_analysis.fourier
       data.mathematical_analysis.correlation
       data.mathematical_analysis.spatial
       data.mathematical_analysis.depth

       Legacy response:

       data.fourier
       data.correlation
       data.spatial
       data.depth
    */


    const analysis =
        data.mathematical_analysis || data;


    if (!analysis || typeof analysis !== "object") {
        return;
    }


    const fourier = analysis.fourier;
    const correlation = analysis.correlation;
    const spatial = analysis.spatial;
    const depth = analysis.depth;


    let visualizationFound = false;


    // FOURIER VISUALIZATION

    if (fourier) {

        const imageURL = getFirstValue(
            fourier,
            ["image", "image_url", "visualization"]
        );

        if (imageURL) {

            setImage("fourierImage", imageURL);

            visualizationFound = true;
        }


        setText(
            "fourierDetails",
            formatModuleDetails(
                fourier.details,
                "Fourier analysis completed."
            )
        );
    }


    // CORRELATION VISUALIZATION

    if (correlation) {

        const imageURL = getFirstValue(
            correlation,
            ["image", "image_url", "visualization"]
        );

        if (imageURL) {

            setImage("correlationImage", imageURL);

            visualizationFound = true;
        }


        setText(
            "correlationDetails",
            formatModuleDetails(
                correlation.details,
                "Correlation analysis completed."
            )
        );
    }


    // SPATIAL VISUALIZATION

    if (spatial) {

        const imageURL = getFirstValue(
            spatial,
            ["image", "image_url", "visualization"]
        );

        if (imageURL) {

            setImage("spatialImage", imageURL);

            visualizationFound = true;
        }


        setText(
            "spatialDetails",
            formatModuleDetails(
                spatial.details,
                "Spatial analysis completed."
            )
        );
    }


    // DEPTH VISUALIZATION

    if (depth) {

        const imageURL = getFirstValue(
            depth,
            ["image", "image_url", "visualization"]
        );

        if (imageURL) {

            setImage("depthImage", imageURL);

            visualizationFound = true;
        }


        setText(
            "depthDetails",
            formatModuleDetails(
                depth.details,
                "Depth estimation completed."
            )
        );
    }


    // Display dashboard only when visualizations exist.

    if (visualizationFound && analysisDashboard) {

        analysisDashboard.style.display = "block";
    }
}


/* ============================================================
 
/* ============================================================
   10. FINAL HYBRID RESULT PANEL
   ============================================================ */

function showFinalResult(result) {

    if (!result) {
        return;
    }


    // Read final hybrid prediction.

    const prediction = getFirstValue(
        result,
        [
            "final_prediction",
            "prediction",
            "predicted_class"
        ]
    );


    // Read hybrid FAKE evidence.

    const fakeEvidence = getEvidencePercentage(
        result,
        [
            "hybrid_fake_evidence_percentage",
            "fake_evidence_percentage",
            "fake_support_percentage",
            "fake_percentage",
            "fake_score"
        ]
    );


    // Read hybrid REAL evidence.

    const realEvidence = getEvidencePercentage(
        result,
        [
            "hybrid_real_evidence_percentage",
            "real_evidence_percentage",
            "real_support_percentage",
            "real_percentage",
            "real_score"
        ]
    );


    if (!prediction) {
        return;
    }


    // Format final classification.

    let resultLabel = String(
        prediction
    ).toUpperCase();


    if (resultLabel.includes("FAKE")) {

        resultLabel = "AI-GENERATED IMAGE (FAKE)";

    } else if (resultLabel.includes("REAL")) {

        resultLabel = "REAL IMAGE";
    }


    setText(
        "resultText",
        resultLabel
    );


    // Display hybrid evidence percentages.

    if (
        Number.isFinite(fakeEvidence) &&
        Number.isFinite(realEvidence)
    ) {

        if (confidenceFill) {

            const safeWidth = Math.max(
                0,
                Math.min(100, fakeEvidence)
            );

            confidenceFill.style.width =
                safeWidth + "%";
        }


        setText(
            "confidenceText",
            "FAKE evidence: " +
            formatPercentage(fakeEvidence) +
            " | REAL evidence: " +
            formatPercentage(realEvidence)
        );

    } else {

        setText(
            "confidenceText",
            "Hybrid evidence values are unavailable in the server response."
        );
    }


    // Apply prediction styling.

    setPredictionColor(
        "resultText",
        prediction
    );


    // Display final result panel.

    if (resultPanel) {

        resultPanel.style.display = "block";
    }
}



/* ============================================================
   11. COMPLETE PREDICTION DASHBOARD
   ============================================================ */

function showPredictionDashboard(data) {

    if (!data || typeof data !== "object") {
        return;
    }


    const individualModels =
        data.individual_models ||
        data.individual_model_results;

    const mathematicalResult =
        data.mathematical_prediction ||
        data.mathematical_result;

    const cnnResult =
        data.cnn_prediction ||
        data.cnn_result;

    const hybridResult =
        data.hybrid_prediction ||
        data.hybrid_result;


    const hasPredictions = Boolean(
        individualModels ||
        mathematicalResult ||
        cnnResult ||
        hybridResult
    );


    if (!hasPredictions) {

        if (predictionDashboard) {
            predictionDashboard.style.display = "none";
        }

        setText(
            "cnnAnalysisStatus",
            "Prediction models have not returned results."
        );

        return;
    }


    /*
       dashboard.js is responsible for rendering:
       - Eight individual models
       - Mathematical ensemble
       - ResNet50 CNN
       - Hybrid prediction
    */

    if (
        window.PixelProofDashboard &&
        typeof window.PixelProofDashboard.render === "function"
    ) {

        window.PixelProofDashboard.render(data);

    } else {

        console.error(
            "PixelProofDashboard is unavailable. " +
            "Check that dashboard.js loads before script.js."
        );
    }


    // Update the separate final result panel.

    if (hybridResult) {

        showFinalResult(hybridResult);
    }
}


/* ============================================================
   12. MAIN IMAGE ANALYSIS REQUEST
   ============================================================ */

if (analyseButton) {

    analyseButton.addEventListener("click", async () => {

        if (!selectedFile) {

            setStatus(
                "Please select an image first."
            );

            return;
        }


        if (analysisInProgress) {
            return;
        }


        const formData = new FormData();

        formData.append(
            "image",
            selectedFile
        );


        analysisInProgress = true;

        setButtonLoading(true);


        setStatus(
            "Running Fourier, correlation, spatial and AI depth analysis. " +
            "The first model run may take longer."
        );


        resetResults();


        try {

            const response = await fetch("/analyze", {

                method: "POST",

                body: formData
            });


            // Parse JSON response.

            let data;

            try {

                data = await response.json();

            } catch {

                throw new Error(
                    "The server returned an invalid response. " +
                    "Check the Flask terminal for errors."
                );
            }


            // Handle HTTP errors.

            if (!response.ok) {

                throw new Error(
                    data.error ||
                    data.message ||
                    `Server error: ${response.status}`
                );
            }


            // Handle unsuccessful analysis.

            if (data.success === false) {

                throw new Error(
                    data.error ||
                    data.message ||
                    "Image analysis was unsuccessful."
                );
            }


            // Display mathematical visualizations.

            showAnalysis(data);


            // Display model predictions when available.

            showPredictionDashboard(data);


            // Determine actual processing status.

            const hasHybridPrediction = Boolean(
                data.hybrid_prediction ||
                data.hybrid_result
            );


            if (
                data.prediction_status === "pending_integration" ||
                !hasHybridPrediction
            ) {

                setStatus(
                    "Image visualizations and depth analysis completed. " +
                    "Prediction models have not yet returned a final hybrid result."
                );

            } else {

                setStatus(
                    "Image analysis and hybrid prediction completed successfully."
                );
            }


            console.log(
                "PixelProof analysis response:",
                data
            );


        } catch (error) {

            console.error(
                "PixelProof analysis error:",
                error
            );


            setStatus(
                "Error: " + error.message
            );


        } finally {

            analysisInProgress = false;

            setButtonLoading(false);
        }
    });
}


/* ============================================================
   13. INITIAL APPLICATION STATE
   ============================================================ */

if (analyseButton) {
    analyseButton.disabled = true;
}


if (resultPanel) {
    resultPanel.style.display = "none";
}


if (analysisDashboard) {
    analysisDashboard.style.display = "none";
}


if (predictionDashboard) {
    predictionDashboard.style.display = "none";
}


console.log(
    "PixelProof Script JavaScript loaded successfully."
);
