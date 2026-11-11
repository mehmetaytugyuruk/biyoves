(() => {
  const variants = ["hair_256", "hair_416", "hair_608"];
  const dropzone = document.querySelector("#dropzone");
  const fileInput = document.querySelector("#file-input");
  const chooseButton = document.querySelector("#choose-button");
  const clearButton = document.querySelector("#clear-button");
  const preview = document.querySelector("#preview");
  const sourceImage = document.querySelector("#source-image");
  const startButton = document.querySelector("#start-button");
  const statusMessage = document.querySelector("#status-message");
  const statusCard = document.querySelector("#status-card");
  const progressTrack = document.querySelector("#progress-track");
  const progressBar = document.querySelector("#progress-bar");
  const resultsSection = document.querySelector("#results-section");
  const presetInputs = [...document.querySelectorAll('input[name="preset"]')];
  const presetLabels = {
    biometric: "Biyometrik",
    vesikalik: "Vesikalık",
    usa_visa: "USA vizesi",
    schengen_visa: "Schengen vizesi",
  };

  let selectedFile = null;
  let selectedPreset = document.querySelector('input[name="preset"]:checked').value;
  let previewUrl = null;
  let activeJob = null;
  let pollTimer = null;
  let progressFrame = null;
  let progressState = {
    actual: 0,
    displayed: 0,
    mode: "idle",
    startedAt: 0,
    completionResolver: null,
  };

  const clampProgress = (progress) => Math.max(0, Math.min(100, Number(progress) || 0));

  const renderProgress = (timestamp) => {
    if (!progressState.startedAt) progressState.startedAt = timestamp;

    const elapsedSeconds = (timestamp - progressState.startedAt) / 1000;
    // Keep the bar gently moving between real server updates, but reserve the
    // final stretch for the actual completed response.
    const gentleProgress = progressState.mode === "active"
      ? Math.min(92, 4 + elapsedSeconds * 0.42)
      : progressState.actual;
    const desiredProgress = Math.max(progressState.actual, gentleProgress);
    const smoothing = progressState.mode === "completing" ? 0.14 : 0.06;
    progressState.displayed += (desiredProgress - progressState.displayed) * smoothing;

    if (progressState.mode === "completing" && progressState.displayed >= 99.5) {
      progressState.displayed = 100;
    }

    progressBar.style.width = `${progressState.displayed}%`;
    progressTrack.setAttribute("aria-valuenow", String(Math.round(progressState.displayed)));

    if (progressState.mode === "completing" && progressState.displayed === 100) {
      const resolve = progressState.completionResolver;
      progressState.completionResolver = null;
      progressFrame = null;
      if (resolve) resolve();
      return;
    }

    progressFrame = window.requestAnimationFrame(renderProgress);
  };

  const ensureProgressFrame = () => {
    if (!progressFrame) progressFrame = window.requestAnimationFrame(renderProgress);
  };

  const setStatus = (message) => {
    statusMessage.textContent = message;
  };

  const resetStartButton = () => {
    startButton.disabled = false;
    startButton.textContent = "İşlemeyi başlat";
  };

  const setPresetOptionsDisabled = (disabled) => {
    presetInputs.forEach((input) => {
      input.disabled = disabled;
      input.closest(".mode-option").classList.toggle("is-disabled", disabled);
    });
  };

  const selectedPresetStatus = () => (
    selectedFile
      ? `${presetLabels[selectedPreset]} seçildi. İşlemeyi başlatabilirsiniz.`
      : "Fotoğraf yükleyin."
  );

  const scrollToResults = () => {
    const behavior = window.matchMedia("(prefers-reduced-motion: reduce)").matches
      ? "auto"
      : "smooth";
    resultsSection.scrollIntoView({ behavior, block: "start" });
  };

  const beginProgress = () => {
    progressState = {
      actual: 2,
      displayed: 0,
      mode: "active",
      startedAt: performance.now(),
      completionResolver: null,
    };
    statusCard.classList.add("is-processing");
    progressTrack.setAttribute("aria-busy", "true");
    progressBar.classList.add("is-active");
    ensureProgressFrame();
  };

  const updateProgress = (progress) => {
    progressState.actual = Math.max(progressState.actual, clampProgress(progress));
    ensureProgressFrame();
  };

  const completeProgress = () => new Promise((resolve) => {
    progressState.mode = "completing";
    progressState.actual = 100;
    progressState.completionResolver = resolve;
    ensureProgressFrame();
  });

  const resetProgress = () => {
    if (progressFrame) window.cancelAnimationFrame(progressFrame);
    progressFrame = null;
    progressState = {
      actual: 0,
      displayed: 0,
      mode: "idle",
      startedAt: 0,
      completionResolver: null,
    };
    statusCard.classList.remove("is-processing");
    progressTrack.setAttribute("aria-busy", "false");
    progressTrack.setAttribute("aria-valuenow", "0");
    progressBar.classList.remove("is-active");
    progressBar.style.width = "0%";
  };

  const clearResults = () => {
    resultsSection.classList.add("is-hidden");
    resultsSection.removeAttribute("data-preset");
    variants.forEach((variant) => {
      document.querySelector(`#result-${variant}`).removeAttribute("src");
      [`download-${variant}`, `print-download-${variant}`].forEach((id) => {
        const link = document.querySelector(`#${id}`);
        link.removeAttribute("href");
        link.classList.add("is-disabled");
        link.setAttribute("aria-disabled", "true");
      });
    });
  };

  const showFile = (file) => {
    if (!file || !file.type.startsWith("image/")) {
      setStatus("Lütfen geçerli bir fotoğraf seçin.");
      return;
    }
    selectedFile = file;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = URL.createObjectURL(file);
    sourceImage.src = previewUrl;
    dropzone.classList.add("is-hidden");
    preview.classList.remove("is-hidden");
    clearButton.classList.remove("is-hidden");
    resetStartButton();
    clearResults();
    resetProgress();
    setStatus("Fotoğraf hazır. İşlemeyi başlatabilirsiniz.");
  };

  const clearPhoto = () => {
    if (activeJob) return;
    selectedFile = null;
    fileInput.value = "";
    if (previewUrl) URL.revokeObjectURL(previewUrl);
    previewUrl = null;
    sourceImage.removeAttribute("src");
    preview.classList.add("is-hidden");
    dropzone.classList.remove("is-hidden");
    clearButton.classList.add("is-hidden");
    startButton.disabled = true;
    startButton.textContent = "İşlemeyi başlat";
    clearResults();
    resetProgress();
    setStatus("Fotoğraf yükleyin.");
  };

  const renderResults = (job) => {
    resultsSection.dataset.preset = job.preset;
    variants.forEach((variant) => {
      const output = job.outputs[variant];
      document.querySelector(`#result-${variant}`).src = `${output.preview_url}?t=${Date.now()}`;
      const link = document.querySelector(`#download-${variant}`);
      link.href = output.download_url;
      link.classList.remove("is-disabled");
      link.removeAttribute("aria-disabled");
      const printLink = document.querySelector(`#print-download-${variant}`);
      printLink.href = output.print_sheet_download_url;
      printLink.classList.remove("is-disabled");
      printLink.removeAttribute("aria-disabled");
    });
    resultsSection.classList.remove("is-hidden");
  };

  const pollJob = async () => {
    if (!activeJob) return;
    try {
      const response = await fetch(`/api/jobs/${activeJob}`);
      const job = await response.json();
      if (!response.ok) throw new Error(job.error || "İş durumu alınamadı.");
      updateProgress(job.progress);
      setStatus(job.status === "queued" ? "Fotoğraf hazırlanıyor…" : "Fotoğraf işleniyor…");
      if (job.status === "completed") {
        setStatus("Çıktılar hazırlanıyor…");
        await completeProgress();
        renderResults(job);
        scrollToResults();
        statusCard.classList.remove("is-processing");
        progressTrack.setAttribute("aria-busy", "false");
        progressBar.classList.remove("is-active");
        setStatus("Fotoğraflar ve çıktılar, sunucu kapandığında silinir.");
        startButton.disabled = true;
        startButton.textContent = "İşlem tamamlandı";
        clearButton.disabled = false;
        setPresetOptionsDisabled(false);
        activeJob = null;
        return;
      }
      if (job.status === "failed") {
        resetProgress();
        resetStartButton();
        clearButton.disabled = false;
        setPresetOptionsDisabled(false);
        activeJob = null;
        setStatus(job.message || "Fotoğraf işlenemedi.");
        return;
      }
      pollTimer = window.setTimeout(pollJob, 500);
    } catch (error) {
      activeJob = null;
      resetProgress();
      resetStartButton();
      clearButton.disabled = false;
      setStatus(error.message || "İş durumu alınamadı.");
    }
  };

  const startProcessing = async () => {
    if (!selectedFile || activeJob) return;
    clearResults();
    startButton.disabled = true;
    clearButton.disabled = true;
    setPresetOptionsDisabled(true);
    beginProgress();
    setStatus("Fotoğraf hazırlanıyor…");
    const data = new FormData();
    data.append("photo", selectedFile);
    data.append("preset", selectedPreset);
    try {
      const response = await fetch("/api/jobs", { method: "POST", body: data });
      const job = await response.json();
      if (!response.ok) throw new Error(job.error || "Fotoğraf yüklenemedi.");
      activeJob = job.id;
      updateProgress(job.progress);
      pollJob();
    } catch (error) {
      resetProgress();
      resetStartButton();
      clearButton.disabled = false;
      setPresetOptionsDisabled(false);
      setStatus(error.message || "Fotoğraf yüklenemedi.");
    }
  };

  chooseButton.addEventListener("click", (event) => { event.stopPropagation(); fileInput.click(); });
  dropzone.addEventListener("click", () => fileInput.click());
  dropzone.addEventListener("keydown", (event) => {
    if (event.key === "Enter" || event.key === " ") fileInput.click();
  });
  fileInput.addEventListener("change", () => showFile(fileInput.files[0]));
  clearButton.addEventListener("click", clearPhoto);
  startButton.addEventListener("click", startProcessing);
  presetInputs.forEach((input) => input.addEventListener("change", () => {
    selectedPreset = input.value;
    presetInputs.forEach((option) => {
      option.closest(".mode-option").classList.toggle("is-selected", option.checked);
    });
    if (!activeJob) {
      resetProgress();
      if (selectedFile) resetStartButton();
      setStatus(selectedPresetStatus());
    }
  }));
  ["dragenter", "dragover"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.add("is-dragging");
  }));
  ["dragleave", "drop"].forEach((eventName) => dropzone.addEventListener(eventName, (event) => {
    event.preventDefault();
    dropzone.classList.remove("is-dragging");
  }));
  dropzone.addEventListener("drop", (event) => showFile(event.dataTransfer.files[0]));
  window.addEventListener("beforeunload", () => {
    if (pollTimer) clearTimeout(pollTimer);
    if (progressFrame) window.cancelAnimationFrame(progressFrame);
  });
})();
