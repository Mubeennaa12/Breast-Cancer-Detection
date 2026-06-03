document.addEventListener("DOMContentLoaded", () => {
    // DOM Elements
    const body = document.body;
    const themeToggle = document.getElementById("theme-toggle");
    const apiStatus = document.getElementById("api-status");
    const pulseDot = document.querySelector(".pulse-dot");
    
    const dropZone = document.getElementById("drop-zone");
    const fileInput = document.getElementById("file-input");
    const analyzeBtn = document.getElementById("analyze-btn");
    
    const overlayColorInput = document.getElementById("overlay-color");
    const colorHexVal = document.querySelector(".color-hex-val");
    const colorDots = document.querySelectorAll(".color-dot");
    const overlayOpacityInput = document.getElementById("overlay-opacity");
    const opacityValLabel = document.getElementById("opacity-val");
    
    const sampleCards = document.querySelectorAll(".sample-card");
    
    const emptyState = document.getElementById("empty-state");
    const loadingState = document.getElementById("loading-state");
    const loadingText = document.getElementById("loading-text");
    const progressFill = document.getElementById("progress-fill");
    const outputDisplay = document.getElementById("output-display");
    const metricsDashboard = document.getElementById("metrics-dashboard");
    
    const visualizerTabs = document.getElementById("visualizer-tabs");
    const tabBtns = document.querySelectorAll(".tab-btn");
    const tabContents = document.querySelectorAll(".tab-content");
    
    const imgOriginalBase = document.getElementById("img-original-base");
    const imgOverlayLayer = document.getElementById("img-overlay-layer");
    const imgCompOrig = document.getElementById("img-comp-orig");
    const imgCompOverlay = document.getElementById("img-comp-overlay");
    const imgMaskOnly = document.getElementById("img-mask-only");
    
    const detectionCard = document.getElementById("detection-card");
    const detectionIcon = document.getElementById("detection-icon");
    const detectionStatus = document.getElementById("detection-status");
    const detectionDesc = document.getElementById("detection-desc");
    const ratioValue = document.getElementById("ratio-value");
    const pixelsValue = document.getElementById("pixels-value");
    
    const downloadOverlayBtn = document.getElementById("download-overlay-btn");
    const downloadMaskBtn = document.getElementById("download-mask-btn");

    let selectedFile = null;
    let analysisResult = null;

    // Check Backend API Health on Load
    checkAPIHealth();

    function checkAPIHealth() {
        fetch("/health")
            .then(res => res.json())
            .then(data => {
                if (data.status === "healthy") {
                    if (data.weights_exist) {
                        apiStatus.textContent = "API: Connected";
                        pulseDot.className = "pulse-dot green";
                    } else {
                        apiStatus.textContent = "API: Weights Missing";
                        pulseDot.className = "pulse-dot";
                        pulseDot.style.backgroundColor = "var(--accent-rose)";
                    }
                } else {
                    apiStatus.textContent = "API: Error";
                    pulseDot.className = "pulse-dot";
                    pulseDot.style.backgroundColor = "var(--accent-rose)";
                }
            })
            .catch(err => {
                apiStatus.textContent = "API: Offline";
                pulseDot.className = "pulse-dot";
                pulseDot.style.backgroundColor = "var(--text-muted)";
            });
    }

    // Theme Toggle Handler
    themeToggle.addEventListener("click", () => {
        body.classList.toggle("light-theme");
        const isLight = body.classList.contains("light-theme");
        themeToggle.innerHTML = isLight ? '<i class="fa-solid fa-sun"></i>' : '<i class="fa-solid fa-moon"></i>';
    });

    // Color controls
    overlayColorInput.addEventListener("input", (e) => {
        const color = e.target.value;
        colorHexVal.textContent = color;
        // Deactivate color dots
        colorDots.forEach(dot => dot.classList.remove("active"));
        
        // If already analyzed, we can re-generate overlay quickly
        if (analysisResult) {
            updateVisualizationSettings();
        }
    });

    colorDots.forEach(dot => {
        dot.addEventListener("click", (e) => {
            e.stopPropagation();
            colorDots.forEach(d => d.classList.remove("active"));
            dot.classList.add("active");
            
            const color = dot.getAttribute("data-color");
            overlayColorInput.value = color;
            colorHexVal.textContent = color;
            
            if (analysisResult) {
                updateVisualizationSettings();
            }
        });
    });

    overlayOpacityInput.addEventListener("input", (e) => {
        const val = parseInt(e.target.value);
        const percent = Math.round((val / 255) * 100);
        opacityValLabel.textContent = `${percent}%`;
        
        if (analysisResult) {
            updateVisualizationSettings();
        }
    });

    // Drag and Drop files
    fileInput.addEventListener("click", (e) => {
        e.stopPropagation();
    });

    dropZone.addEventListener("click", () => {
        fileInput.click();
    });

    dropZone.addEventListener("dragover", (e) => {
        e.preventDefault();
        dropZone.classList.add("dragging");
    });

    dropZone.addEventListener("dragleave", () => {
        dropZone.classList.remove("dragging");
    });

    dropZone.addEventListener("drop", (e) => {
        e.preventDefault();
        dropZone.classList.remove("dragging");
        if (e.dataTransfer.files.length > 0) {
            handleFileSelect(e.dataTransfer.files[0]);
        }
    });

    fileInput.addEventListener("change", (e) => {
        if (e.target.files.length > 0) {
            handleFileSelect(e.target.files[0]);
        }
    });

    function handleFileSelect(file) {
        if (!file.type.startsWith("image/")) {
            alert("Please select a valid image file (scan).");
            return;
        }
        selectedFile = file;
        
        // Update drag zone content with file name
        const contentDiv = dropZone.querySelector(".drag-zone-content");
        contentDiv.innerHTML = `
            <i class="fa-solid fa-file-image drag-icon" style="color: var(--accent-cyan)"></i>
            <h4>Selected File</h4>
            <p>${file.name} (${(file.size / 1024).toFixed(1)} KB)</p>
            <span class="file-info-limit">Click or drop a different file to change</span>
        `;
        
        // Deactivate sample cards
        sampleCards.forEach(card => card.classList.remove("active"));
        
        analyzeBtn.removeAttribute("disabled");
    }

    // Predefined Samples trigger
    sampleCards.forEach(card => {
        card.addEventListener("click", () => {
            sampleCards.forEach(c => c.classList.remove("active"));
            card.classList.add("active");
            
            const sampleName = card.getAttribute("data-sample");
            const sampleUrl = `samples/${sampleName}`;
            
            // Set drag zone content
            const contentDiv = dropZone.querySelector(".drag-zone-content");
            contentDiv.innerHTML = `
                <i class="fa-solid fa-file-medical drag-icon" style="color: var(--accent-cyan)"></i>
                <h4>Sample Loaded</h4>
                <p>${sampleName}</p>
                <span class="file-info-limit">Demo scan loaded</span>
            `;
            
            // Fetch the image file to convert to a blob
            fetch(sampleUrl)
                .then(res => {
                    if (!res.ok) throw new Error("Sample image not found on server");
                    return res.blob();
                })
                .then(blob => {
                    selectedFile = new File([blob], sampleName, { type: "image/png" });
                    analyzeBtn.removeAttribute("disabled");
                    // Automatically trigger analysis for seamless demo
                    analyzeImage();
                })
                .catch(err => {
                    console.error("Error loading sample:", err);
                    alert("Could not load sample scan from server. Please upload your own image.");
                });
        });
    });

    // Execute Inference API call
    analyzeBtn.addEventListener("click", () => {
        analyzeImage();
    });

    function analyzeImage() {
        if (!selectedFile) return;

        // Reset UI display
        emptyState.classList.add("hidden");
        outputDisplay.classList.add("hidden");
        metricsDashboard.classList.add("hidden");
        loadingState.classList.remove("hidden");
        
        // Start simulated progress steps
        let progress = 0;
        progressFill.style.width = "0%";
        
        const steps = [
            { limit: 25, text: "Ingesting scan and resizing to 256x256..." },
            { limit: 50, text: "Running ImageNet normalizations..." },
            { limit: 75, text: "Executing Multi-Level Fusion Model..." },
            { limit: 95, text: "Decoding logits and rendering mask overlay..." }
        ];
        
        let stepIdx = 0;
        const progressInterval = setInterval(() => {
            if (stepIdx < steps.length) {
                const currentStep = steps[stepIdx];
                if (progress < currentStep.limit) {
                    progress += Math.floor(Math.random() * 5) + 2;
                    if (progress > currentStep.limit) progress = currentStep.limit;
                    progressFill.style.width = `${progress}%`;
                    loadingText.textContent = currentStep.text;
                } else {
                    stepIdx++;
                }
            }
        }, 50);

        // Prepare API request payload
        const formData = new FormData();
        formData.append("image", selectedFile);
        formData.append("color", overlayColorInput.value);
        formData.append("opacity", overlayOpacityInput.value);

        fetch("/predict", {
            method: "POST",
            body: formData
        })
        .then(res => res.json())
        .then(data => {
            clearInterval(progressInterval);
            if (data.error) {
                throw new Error(data.error);
            }
            
            // Finish progress bar
            progressFill.style.width = "100%";
            
            setTimeout(() => {
                analysisResult = data;
                renderResults(data);
            }, 300);
        })
        .catch(err => {
            clearInterval(progressInterval);
            console.error("Error:", err);
            loadingState.classList.add("hidden");
            emptyState.classList.remove("hidden");
            alert(`Analysis failed: ${err.message}`);
        });
    }

    function renderResults(data) {
        loadingState.classList.add("hidden");
        outputDisplay.classList.remove("hidden");
        metricsDashboard.classList.remove("hidden");

        // Set Image sources
        imgOriginalBase.src = data.images.original;
        imgOverlayLayer.src = data.images.overlay;
        imgCompOrig.src = data.images.original;
        imgCompOverlay.src = data.images.overlay;
        imgMaskOnly.src = data.images.mask;

        // Render diagnosis metrics card
        if (data.detected) {
            detectionCard.className = "metric-card tumor";
            detectionIcon.className = "fa-solid fa-triangle-exclamation";
            detectionStatus.textContent = "TUMOR DETECTED";
            detectionDesc.textContent = "Abnormal tissue segmented. Clinical review is highly recommended.";
        } else {
            detectionCard.className = "metric-card normal";
            detectionIcon.className = "fa-solid fa-circle-check";
            detectionStatus.textContent = "NO TUMOR DETECTED";
            detectionDesc.textContent = "No significant anomalous mass segmented. Scan matches normal profile.";
        }

        // Render numbers
        ratioValue.textContent = `${data.tumor_percentage.toFixed(1)}%`;
        pixelsValue.textContent = `${data.tumor_pixels.toLocaleString()} px`;
    }

    // Local adjustment of opacity and color via API refresh
    function updateVisualizationSettings() {
        if (!selectedFile || !analysisResult) return;
        
        // Fast local update of opacity without making API calls for opacity slider adjustments
        // Because overlay-layer is an absolute image, we can adjust its CSS opacity!
        const opacity = parseFloat(overlayOpacityInput.value) / 255;
        imgOverlayLayer.style.opacity = opacity;
        
        // But for tint color changes, we trigger a request to server to fetch new overlay
        // We throttle color picker changes so we don't spam requests
        clearTimeout(window.colorTimeout);
        window.colorTimeout = setTimeout(() => {
            const formData = new FormData();
            formData.append("image", selectedFile);
            formData.append("color", overlayColorInput.value);
            formData.append("opacity", overlayOpacityInput.value);
            
            fetch("/predict", {
                method: "POST",
                body: formData
            })
            .then(res => res.json())
            .then(data => {
                if (data.images && data.images.overlay) {
                    imgOverlayLayer.src = data.images.overlay;
                    imgCompOverlay.src = data.images.overlay;
                }
            })
            .catch(err => console.error("Error updating overlay tint:", err));
        }, 150);
    }

    // Tabs switching
    tabBtns.forEach(btn => {
        btn.addEventListener("click", () => {
            tabBtns.forEach(b => b.classList.remove("active"));
            btn.classList.add("active");
            
            const tabName = btn.getAttribute("data-tab");
            tabContents.forEach(content => {
                if (content.id === `tab-${tabName}`) {
                    content.classList.remove("hidden");
                } else {
                    content.classList.add("hidden");
                }
            });
        });
    });

    // Download handlers
    downloadOverlayBtn.addEventListener("click", () => {
        if (!analysisResult) return;
        triggerDownload(imgOverlayLayer.src, "scan_tumor_overlay.png");
    });

    downloadMaskBtn.addEventListener("click", () => {
        if (!analysisResult) return;
        triggerDownload(imgMaskOnly.src, "scan_tumor_mask.png");
    });

    function triggerDownload(dataUrl, filename) {
        const link = document.createElement("a");
        link.href = dataUrl;
        link.download = filename;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);
    }
});
