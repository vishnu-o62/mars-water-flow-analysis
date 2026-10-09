import { useEffect, useState } from "react";
import "./App.css";

const API_BASE_URL = "http://127.0.0.1:5000";
const MAX_UPLOAD_SIZE = 5 * 1024 * 1024;
const ALLOWED_EXTENSIONS = new Set([".jpg", ".jpeg", ".png"]);
const ALLOWED_MIME_TYPES = new Set(["image/jpeg", "image/png", "image/jpg"]);

type AnalysisResult = {
  success: boolean;
  image: {
    name: string;
    url: string;
  };
  images: {
    original: string;
    prediction: string;
    cleaned: string;
    valley: string;
    skeleton: string;
    direction: string;
    reconstruction: string;
  };
  summary: {
    valleys_detected: number;
    flow_paths: number;
    accepted_paths?: number;
    skeleton_pixels: number | null;
    best_path_length: number | null;
    best_path_confidence: number | null;
    processing_time: number;
    analysis_status: string;
  };
  metrics?: {
    accuracy?: number;
    precision?: number;
    recall?: number;
    f1?: number;
    iou?: number;
  };
  pipeline: {
    model_used: string;
    sam2_mask: string;
    prediction: string;
    cleaned_mask: string;
    valley_detection: string;
    skeleton: string;
    flow_direction: string;
    flow_reconstruction: string;
  };
};

type GalleryRun = {
  id: string;
  imageName: string;
  thumbnail: string | null;
  completedAt: string;
  result: AnalysisResult;
};

const GALLERY_STORAGE_KEY = "mars-water-flow-gallery-v1";
const MAX_GALLERY_RUNS = 20;

type PanelProps = {
  step: number;
  title: string;
  subtitle?: string;
  image?: string;
  fallbackImage?: string;
  showNoFeatureOverlay?: boolean;
  description: string;
  loading: boolean;
  resultReady: boolean;
  accent: string;
  onPreview: (src: string, title: string) => void;
};

function Loader({ compact = false }: { compact?: boolean }) {
  return (
    <span className={`loader ${compact ? "loader-compact" : ""}`}>
      <i />
      <i />
      <i />
    </span>
  );
}

function Panel({
  step,
  title,
  subtitle,
  image,
  fallbackImage,
  showNoFeatureOverlay = false,
  description,
  loading,
  resultReady,
  accent,
  onPreview,
}: PanelProps) {
  return (
    <article className="pipeline-card">
      <div className="pipeline-card-header">
        <span className="step-badge" style={{ "--step-accent": accent } as React.CSSProperties}>
          {step}
        </span>

        <div>
          <h3>{title}</h3>
          {subtitle && <span>{subtitle}</span>}
        </div>
      </div>

      <div className="pipeline-image-shell">
        {loading ? (
          <div className="image-loading">
            <Loader />
            <span>PROCESSING</span>
          </div>
        ) : resultReady && (image || fallbackImage) ? (
          <>
            <img
              src={`${API_BASE_URL}${showNoFeatureOverlay && fallbackImage ? fallbackImage : image}`}
              alt={title}
            />

            {showNoFeatureOverlay && fallbackImage && (
              <div className="no-feature-overlay" aria-label="No features detected">
                <span className="no-feature-symbol" aria-hidden="true">⊘</span>
                <strong>No features detected</strong>
              </div>
            )}

            <button
              className="expand-image"
              type="button"
              onClick={() =>
                onPreview(
                  `${API_BASE_URL}${showNoFeatureOverlay && fallbackImage ? fallbackImage : image}`,
                  title
                )
              }
              aria-label={`Preview ${title}`}
              title="Preview image"
            >
              ↗
            </button>
          </>
        ) : (
          <div className="image-empty">
            <span>◌</span>
            <small>Output will appear here</small>
          </div>
        )}
      </div>

      <p className="pipeline-description">{description}</p>
    </article>
  );
}

function StatCard({
  icon,
  value,
  label,
  tone,
}: {
  icon: string;
  value: string;
  label: string;
  tone: string;
}) {
  return (
    <div className={`stat-card ${tone}`}>
      <span className="stat-icon">{icon}</span>
      <div>
        <strong>{value}</strong>
        <small>{label}</small>
      </div>
    </div>
  );
}

function DetailRow({ label, value }: { label: string; value: string }) {
  return (
    <div className="detail-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function MetricCard({ label, value }: { label: string; value?: number }) {
  return (
    <div className="metric-card">
      <span>{label}</span>
      <strong>{typeof value === "number" ? `${value.toFixed(2)}%` : "N/A"}</strong>
    </div>
  );
}

function createGalleryThumbnail(file: File): Promise<string | null> {
  return new Promise((resolve) => {
    const reader = new FileReader();

    reader.onload = () => {
      const image = new Image();
      image.onload = () => {
        const maxWidth = 480;
        const maxHeight = 320;
        const scale = Math.min(maxWidth / image.width, maxHeight / image.height, 1);
        const canvas = document.createElement("canvas");
        canvas.width = Math.max(1, Math.round(image.width * scale));
        canvas.height = Math.max(1, Math.round(image.height * scale));

        const context = canvas.getContext("2d");
        if (!context) {
          resolve(null);
          return;
        }

        context.drawImage(image, 0, 0, canvas.width, canvas.height);
        resolve(canvas.toDataURL("image/jpeg", 0.72));
      };
      image.onerror = () => resolve(null);
      image.src = String(reader.result);
    };

    reader.onerror = () => resolve(null);
    reader.readAsDataURL(file);
  });
}

function formatGalleryDate(value: string) {
  const date = new Date(value);
  return {
    date: date.toLocaleDateString(undefined, {
      day: "2-digit",
      month: "long",
      year: "numeric",
    }),
    time: date.toLocaleTimeString(undefined, {
      hour: "2-digit",
      minute: "2-digit",
      second: "2-digit",
    }),
  };
}

function App() {
  const [selectedImage, setSelectedImage] = useState<File | null>(null);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [darkMode, setDarkMode] = useState(true);
  const [showMetrics, setShowMetrics] = useState(false);
  const [preview, setPreview] = useState<{ src: string; title: string } | null>(null);
  const [marsSuitabilityWarning, setMarsSuitabilityWarning] = useState(false);
  const [completedAt, setCompletedAt] = useState<Date | null>(null);
  const [galleryRuns, setGalleryRuns] = useState<GalleryRun[]>(() => {
    try {
      const saved = localStorage.getItem(GALLERY_STORAGE_KEY);
      if (!saved) return [];
      const parsed = JSON.parse(saved);
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  });

  useEffect(() => {
    try {
      localStorage.setItem(GALLERY_STORAGE_KEY, JSON.stringify(galleryRuns));
    } catch (storageError) {
      console.warn("Gallery history could not be saved locally.", storageError);
    }
  }, [galleryRuns]);

  const handleImageSelect = (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0] ?? null;

    setResult(null);
    setError("");
    setShowMetrics(false);
    setMarsSuitabilityWarning(false);
    setCompletedAt(null);

    if (!file) {
      setSelectedImage(null);
      return;
    }

    const extension = file.name.includes(".")
      ? `.${file.name.split(".").pop()?.toLowerCase()}`
      : "";

    if (!ALLOWED_EXTENSIONS.has(extension)) {
      setSelectedImage(null);
      event.target.value = "";
      setError("Unsupported file type. Please select a JPG, JPEG, or PNG image.");
      return;
    }

    if (file.size > MAX_UPLOAD_SIZE) {
      setSelectedImage(null);
      event.target.value = "";
      setError("The selected file is too large. Maximum allowed size is 5 MB.");
      return;
    }

    // If the browser provides a MIME type, reject an explicitly incompatible one.
    // The backend performs the authoritative content validation.
    if (file.type && !ALLOWED_MIME_TYPES.has(file.type.toLowerCase())) {
      setSelectedImage(null);
      event.target.value = "";
      setError("Unsupported file type. Please select a JPG, JPEG, or PNG image.");
      return;
    }

    setMarsSuitabilityWarning(true);
    setSelectedImage(file);
  };

  const handleAnalyze = async () => {
    if (!selectedImage) {
      setError("Please select a Mars image first.");
      return;
    }

    setLoading(true);
    setError("");
    setResult(null);
    setShowMetrics(false);
    setCompletedAt(null);

    try {
      const formData = new FormData();
      formData.append("image", selectedImage);

      const response = await fetch(`${API_BASE_URL}/api/analyze`, {
        method: "POST",
        body: formData,
      });

      const data = await response.json().catch(() => null);

      if (!response.ok || !data?.success) {
        const apiError = data?.error;
        const message =
          typeof apiError === "object" && apiError !== null && "message" in apiError
            ? String(apiError.message)
            : typeof apiError === "string"
              ? apiError
              : "Analysis could not be completed.";

        throw new Error(message);
      }

      const completionTime = new Date();
      setResult(data);
      setCompletedAt(completionTime);

      const thumbnail = await createGalleryThumbnail(selectedImage);
      const galleryRun: GalleryRun = {
        id: `${completionTime.getTime()}-${Math.random().toString(36).slice(2, 8)}`,
        imageName: selectedImage.name,
        thumbnail,
        completedAt: completionTime.toISOString(),
        result: data,
      };

      setGalleryRuns((currentRuns) => [galleryRun, ...currentRuns].slice(0, MAX_GALLERY_RUNS));
    } catch (err) {
      console.error(err);
      setError(err instanceof Error ? err.message : "Unable to connect to Flask.");
    } finally {
      setLoading(false);
    }
  };

  const valleys = result?.summary.valleys_detected ?? null;
  const flowPaths = result?.summary.flow_paths ?? null;
  const acceptedPaths = result?.summary.accepted_paths ?? null;
  const skeletonPixels = result?.summary.skeleton_pixels ?? null;
  const bestPathLength = result?.summary.best_path_length;
  const bestConfidence = result?.summary.best_path_confidence;
  const processingTime = result?.summary.processing_time;

  const pathText = (count: number | null) =>
    count === null ? "—" : `${count} ${count === 1 ? "path" : "paths"}`;

  const reconstructionText = (count: number | null) =>
    count === null
      ? result?.pipeline.flow_reconstruction || "Waiting"
      : `${count} ${count === 1 ? "path" : "paths"} reconstructed`;

  const metricsAvailable = Boolean(
    result?.metrics &&
      Object.values(result.metrics).some((value) => typeof value === "number")
  );

  // Pipeline 1 outputs remain visible even when no downstream geological feature
  // is detected. Downstream stages use the original image as a visual fallback
  // with a clear no-feature overlay instead of showing a blank/black panel.
  const hasValleys = (valleys ?? 0) > 0;
  const hasSkeleton = (skeletonPixels ?? 0) > 0;
  const hasFlow = (flowPaths ?? 0) > 0 || (acceptedPaths ?? 0) > 0;

  return (
    <div className={`mars-site ${darkMode ? "dark" : "light"}`}>
      <aside className="sidebar">
        <div className="sidebar-brand">
          <div className="mini-mars" aria-hidden="true">
            <span />
            <span />
            <span />
          </div>

          <div>
            <strong>MARS</strong>
            <span>Water-Flow Analysis</span>
            <small>AI for a Wetter Tomorrow</small>
          </div>
        </div>

        <nav className="sidebar-nav" aria-label="Primary navigation">
          <a href="#home">
            <span>⌂</span>
            Home
          </a>
          <a href="#analysis" className="active">
            <span>▣</span>
            Analyze Image
          </a>
          <a href="#summary">
            <span>▥</span>
            Results
          </a>
          <a href="#metrics">
            <span>⌁</span>
            Metrics
          </a>
          <a href="#gallery">
            <span>▦</span>
            Gallery
          </a>
          <a href="#about">
            <span>ⓘ</span>
            About
          </a>
        </nav>

        <div className="sidebar-quote">
          <span>“Exploring</span>
          <span>Mars' Past</span>
          <span>for a Wetter</span>
          <span>Tomorrow”</span>
        </div>

        <button
          className="sidebar-theme"
          type="button"
          onClick={() => setDarkMode((value) => !value)}
        >
          <span className="theme-symbol">{darkMode ? "☼" : "☾"}</span>
          <span>{darkMode ? "Dark Mode" : "Light Mode"}</span>
          <i className={darkMode ? "toggle on" : "toggle"}><b /></i>
        </button>
      </aside>

      <div className="site-main">
        <header className="hero" id="home">
          <div className="hero-stars" />
          <div className="hero-mountains" />

          <div className="hero-copy">
            <div className="hero-kicker">
              <span>PLANETARY AI</span>
              <span>•</span>
              <span>GEOLOGICAL ANALYSIS</span>
            </div>

            <h1>
              AI-Based Identification of
              <br />
              Past <em>Water Flow</em> on Mars
            </h1>

            <p>
              Deep learning and geospatial analysis to detect and reconstruct
              potential ancient water flow paths on the Martian surface.
            </p>

            <div className="hero-features">
              <span><b>◉</b> AI Powered</span>
              <span><b>⌁</b> Geological Analysis</span>
              <span><b>✥</b> Research Driven</span>
            </div>
          </div>

          <div className="hero-planet" aria-hidden="true">
            <div className="planet-surface" />
            <div className="planet-crater c1" />
            <div className="planet-crater c2" />
            <div className="planet-crater c3" />
          </div>

          <blockquote className="hero-quote">
            “Uncovering the traces
            <br />
            of water, to understand
            <br />
            a wetter Mars.”
            <span />
          </blockquote>
        </header>

        <main className="content">
          <section className="control-bar" id="analysis">
            <div className="selected-file">
              <div className="file-preview">
                {selectedImage ? (
                  <span>{selectedImage.name.split(".").pop()?.toUpperCase()}</span>
                ) : (
                  <span>MARS</span>
                )}
              </div>

              <div>
                <small>Selected Mars Image</small>
                <strong>{selectedImage?.name || result?.image.name || "No image selected"}</strong>
              </div>

              <label className="change-image">
                <span>□</span>
                {selectedImage ? "Change Image" : "Select Image"}
                <input
                  type="file"
                  accept="image/*"
                  onChange={handleImageSelect}
                  disabled={loading}
                />
              </label>
            </div>

            <button
              className="run-button"
              type="button"
              onClick={handleAnalyze}
              disabled={!selectedImage || loading}
            >
              {loading ? <Loader compact /> : <span>▶</span>}
              {loading ? "Analyzing..." : "Run Analysis"}
            </button>

            <div className={`completion-card ${loading ? "processing" : result ? "complete" : ""}`}>
              <div className="completion-icon">
                {loading ? <Loader compact /> : result ? "✓" : "•"}
              </div>
              <div className="completion-main">
                <strong>
                  {loading ? "Analysis In Progress" : result ? "Analysis Completed" : "System Ready"}
                </strong>
                <small>
                  {loading
                    ? "Processing selected Mars image"
                    : result
                      ? `Processed in ${processingTime?.toFixed(1)} seconds`
                      : "Select an image to begin"}
                </small>
              </div>

              {completedAt && result && (
                <div className="completion-timestamp" aria-label="Analysis completion date and time">
                  <span>COMPLETED</span>
                  <strong>{completedAt.toLocaleDateString(undefined, {
                    day: "2-digit",
                    month: "long",
                    year: "numeric",
                  })}</strong>
                  <small>{completedAt.toLocaleTimeString(undefined, {
                    hour: "2-digit",
                    minute: "2-digit",
                    second: "2-digit",
                  })}</small>
                </div>
              )}
            </div>
          </section>

          {error && (
            <div className="error-message" role="alert">
              <b>!</b>
              {error}
            </div>
          )}

          {marsSuitabilityWarning && !error && (
            <div className="mars-suitability-warning" role="note">
              <b>ⓘ</b>
              <div>
                <strong>Image Suitability Reminder</strong>
                <span>
                  The uploaded image may not be suitable for Mars water-flow analysis.
                  Please upload a Mars surface image, preferably from a NASA
                  planetary-imaging dataset.
                </span>
              </div>
            </div>
          )}

          {loading && (
            <section className="analysis-loader" aria-live="polite">
              <div className="loader-orbit">
                <span />
              </div>
              <div className="loader-main">
                <span className="hero-kicker">AI PIPELINE ACTIVE</span>
                <h2>Analyzing Mars image...</h2>
                <p>
                  Preparing the water-flow reconstruction outputs.
                </p>
                <div className="loading-line">
                  <span />
                </div>
                <div className="loading-stages">
                  <b>U-Net</b>
                  <i>→</i>
                  <b>Post-processing</b>
                  <i>→</i>
                  <b>Valleys</b>
                  <i>→</i>
                  <b>Skeleton</b>
                  <i>→</i>
                  <span>Reconstruction</span>
                </div>
              </div>
            </section>
          )}

          <section className="workspace">
            <div className="workspace-heading">
              <div>
                <span className="section-label">ANALYSIS PIPELINE</span>
                <h2>Analysis Pipeline Results</h2>
                <p>Step-by-step visualization of the processing pipeline</p>
              </div>

              <div className={`workspace-status ${loading ? "processing" : result ? "complete" : ""}`}>
                <span />
                {loading ? "Analysis in progress" : result ? "Analysis completed" : "Awaiting analysis"}
              </div>
            </div>

            <div className="pipeline-grid">
              <Panel
                step={1}
                title="Original Mars Image"
                image={result?.images.original}
                description={`Input image from Mars dataset${result ? ` (${result.image.name})` : ""}`}
                loading={loading}
                resultReady={Boolean(result)}
                accent="#2d7cff"
                onPreview={(src, title) => setPreview({ src, title })}
              />

              <Panel
                step={2}
                title="U-Net Predicted Mask"
                subtitle="Raw U-Net Prediction"
                image={result?.images.prediction}
                description="Raw prediction from trained U-Net model"
                loading={loading}
                resultReady={Boolean(result)}
                accent="#8252ff"
                onPreview={(src, title) => setPreview({ src, title })}
              />

              <Panel
                step={3}
                title="Cleaned Prediction Mask"
                subtitle="After Post-processing"
                image={result?.images.cleaned}
                description="Refined binary mask after post-processing"
                loading={loading}
                resultReady={Boolean(result)}
                accent="#19b86b"
                onPreview={(src, title) => setPreview({ src, title })}
              />

              <Panel
                step={4}
                title="Valley Detection"
                image={result?.images.valley}
                fallbackImage={result?.images.original}
                showNoFeatureOverlay={Boolean(result) && !hasValleys}
                description="Detected valley regions / candidate areas"
                loading={loading}
                resultReady={Boolean(result)}
                accent="#ff793f"
                onPreview={(src, title) => setPreview({ src, title })}
              />

              <Panel
                step={5}
                title="Skeleton / Centerline"
                image={result?.images.skeleton}
                fallbackImage={result?.images.original}
                showNoFeatureOverlay={Boolean(result) && !hasSkeleton}
                description="Extracted skeleton / centerline of valleys"
                loading={loading}
                resultReady={Boolean(result)}
                accent="#ff4c4c"
                onPreview={(src, title) => setPreview({ src, title })}
              />

              <Panel
                step={6}
                title="Flow Direction"
                image={result?.images.direction}
                fallbackImage={result?.images.original}
                showNoFeatureOverlay={Boolean(result) && !hasFlow}
                description="Estimated flow direction on skeleton"
                loading={loading}
                resultReady={Boolean(result)}
                accent="#227cff"
                onPreview={(src, title) => setPreview({ src, title })}
              />

              <Panel
                step={7}
                title="Flow Reconstruction"
                image={result?.images.reconstruction}
                fallbackImage={result?.images.original}
                showNoFeatureOverlay={Boolean(result) && !hasFlow}
                description="Reconstructed water flow paths on Mars surface"
                loading={loading}
                resultReady={Boolean(result)}
                accent="#a64cff"
                onPreview={(src, title) => setPreview({ src, title })}
              />
            </div>
          </section>

          <section className="gallery-section" id="gallery">
            <div className="section-title gallery-heading">
              <div>
                <span className="section-label">RUN HISTORY</span>
                <h2>Analysis Gallery</h2>
                <p>Saved locally in this browser. Up to 20 recent analysis runs are retained.</p>
              </div>

              {galleryRuns.length > 0 && (
                <button
                  className="gallery-clear-button"
                  type="button"
                  onClick={() => {
                    if (window.confirm("Clear all saved analysis runs from this browser?")) {
                      setGalleryRuns([]);
                    }
                  }}
                >
                  Clear Gallery
                </button>
              )}
            </div>

            {galleryRuns.length === 0 ? (
              <div className="gallery-empty">
                <span>▦</span>
                <strong>No analysis runs saved yet</strong>
                <small>Completed analyses will appear here automatically.</small>
              </div>
            ) : (
              <div className="gallery-grid">
                {galleryRuns.map((run) => {
                  const timestamp = formatGalleryDate(run.completedAt);
                  const runValleys = run.result.summary.valleys_detected;
                  const runFlows = run.result.summary.flow_paths;
                  const runAccepted = run.result.summary.accepted_paths;

                  return (
                    <article className="gallery-card" key={run.id}>
                      <div className="gallery-thumbnail">
                        {run.thumbnail ? (
                          <img src={run.thumbnail} alt={`Saved preview of ${run.imageName}`} />
                        ) : (
                          <span>IMAGE</span>
                        )}
                      </div>

                      <div className="gallery-card-body">
                        <div className="gallery-card-top">
                          <strong title={run.imageName}>{run.imageName}</strong>
                          <span>✓ Completed</span>
                        </div>

                        <div className="gallery-time">
                          <b>{timestamp.date}</b>
                          <span>{timestamp.time}</span>
                        </div>

                        <div className="gallery-stats">
                          <span><b>{runValleys}</b> Valleys</span>
                          <span><b>{runFlows}</b> Flows</span>
                          <span><b>{runAccepted ?? "—"}</b> Paths</span>
                          <span><b>{run.result.summary.processing_time.toFixed(1)}s</b> Process</span>
                        </div>

                        <button
                          className="gallery-view-button"
                          type="button"
                          onClick={() => {
                            setResult(run.result);
                            setCompletedAt(new Date(run.completedAt));
                            setSelectedImage(null);
                            setShowMetrics(false);
                            setError("");
                            document.getElementById("summary")?.scrollIntoView({ behavior: "smooth", block: "start" });
                          }}
                        >
                          View Results →
                        </button>
                      </div>
                    </article>
                  );
                })}
              </div>
            )}
          </section>

          <section className="results-layout" id="summary">
            <div className="details-column">
              <div className="section-title">
                <span className="section-label">RESULTS</span>
                <h2>Analysis Details</h2>
              </div>

              <div className="details-card">
                <DetailRow label="Image Name" value={result?.image.name ?? "—"} />
                <DetailRow label="Valleys Detected" value={valleys !== null ? String(valleys) : "—"} />
                <DetailRow label="Reconstructed Flows" value={flowPaths !== null ? String(flowPaths) : "—"} />
                <DetailRow label="Accepted Paths" value={acceptedPaths !== null ? String(acceptedPaths) : "—"} />
                <DetailRow label="Skeleton Pixels" value={skeletonPixels !== null ? skeletonPixels.toLocaleString() : "—"} />
                <DetailRow
                  label="Best Path Length"
                  value={bestPathLength !== null && bestPathLength !== undefined ? `${bestPathLength.toFixed(1)} px` : "N/A"}
                />
                <DetailRow
                  label="Best Path Confidence"
                  value={bestConfidence !== null && bestConfidence !== undefined ? bestConfidence.toFixed(2) : "N/A"}
                />
                <DetailRow
                  label="Processing Time"
                  value={processingTime !== undefined ? `${processingTime.toFixed(1)} seconds` : "—"}
                />
                <DetailRow label="Analysis Status" value={result?.summary.analysis_status ?? "Waiting"} />
              </div>
            </div>

            <div className="summary-column">
              <div className="section-title">
                <span className="section-label">DETECTION SUMMARY</span>
                <h2>Detected Features</h2>
              </div>

              <div className="stats-grid">
                <StatCard
                  icon="⌁"
                  value={valleys !== null ? String(valleys) : "—"}
                  label="Valleys Detected"
                  tone="orange"
                />
                <StatCard
                  icon="≋"
                  value={flowPaths !== null ? String(flowPaths) : "—"}
                  label="Reconstructed Flow Paths"
                  tone="blue"
                />
                <StatCard
                  icon="⌘"
                  value={acceptedPaths !== null ? String(acceptedPaths) : "—"}
                  label="Accepted Paths"
                  tone="blue"
                />
                <StatCard
                  icon="◷"
                  value={processingTime !== undefined ? `${processingTime.toFixed(1)} s` : "—"}
                  label="Processing Time"
                  tone="blue"
                />
              </div>

              <div className="flow-count-card">
                <div>
                  <span>FLOW RECONSTRUCTION</span>
                  <strong>{reconstructionText(flowPaths)}</strong>
                </div>
                <div className="flow-count-icon">≋</div>
              </div>
            </div>

            <div className="actions-column" id="metrics">
              <div className="section-title">
                <span className="section-label">EVALUATION</span>
                <h2>Model Evaluation</h2>
              </div>

              <button
                className="metrics-button"
                type="button"
                disabled={!result}
                onClick={() => setShowMetrics(true)}
              >
                <span className="metrics-icon">▥</span>
                <span>
                  <strong>View Metrics</strong>
                  <small>
                    {metricsAvailable
                      ? "See evaluation metrics for this image"
                      : "Evaluation metrics for this image"}
                  </small>
                </span>
                <b>›</b>
              </button>

              <div className="interpretation-card">
                <span className="interpretation-icon">💡</span>
                <div>
                  <h3>Interpretation</h3>
                  <p>
                    {result
                      ? valleys && flowPaths
                        ? "Multiple valley structures and flow paths have been detected, indicating potential geological features associated with past water flow on Mars."
                        : "The analysis has completed. Review the detected features and reconstructed paths above."
                      : "Run an analysis to receive an interpretation of the detected geological features."}
                  </p>
                </div>
              </div>
            </div>
          </section>

          <section className="pipeline-info" id="about">
            <div>
              <span className="section-label">SYSTEM</span>
              <h2>AI water-flow analysis</h2>
              <p>
                The interface presents the outputs of the existing analysis
                pipeline without changing its underlying processing stages.
              </p>
            </div>

            <div className="pipeline-info-items">
              <span><b>01</b> U-Net Segmentation</span>
              <span><b>02</b> Valley Detection</span>
              <span><b>03</b> Skeleton Extraction</span>
              <span><b>04</b> Flow Reconstruction</span>
            </div>
          </section>
        </main>

        <footer className="site-footer">
          <span>Mars Water-Flow Analysis</span>
          <span>AI for Planetary Exploration</span>
          <div>
            <span>Research</span>
            <i>•</i>
            <span>Explore</span>
            <i>•</i>
            <span>Discover</span>
            <i>•</i>
            <span>A Wetter Tomorrow</span>
          </div>
        </footer>
      </div>

      {showMetrics && result && (
        <div className="modal-backdrop" onClick={() => setShowMetrics(false)}>
          <div
            className="metrics-modal"
            onClick={(event) => event.stopPropagation()}
            role="dialog"
            aria-modal="true"
            aria-labelledby="metrics-title"
          >
            <div className="modal-top">
              <div>
                <span className="section-label">MODEL EVALUATION</span>
                <h2 id="metrics-title">Evaluation Metrics</h2>
                <p>Evaluation values returned by the analysis backend.</p>
              </div>

              <button type="button" className="close-modal" onClick={() => setShowMetrics(false)}>
                ×
              </button>
            </div>

            <div className="metrics-grid">
              <MetricCard label="Accuracy" value={result.metrics?.accuracy} />
              <MetricCard label="Precision" value={result.metrics?.precision} />
              <MetricCard label="Recall" value={result.metrics?.recall} />
              <MetricCard label="F1 Score" value={result.metrics?.f1} />
              <MetricCard label="IoU" value={result.metrics?.iou} />
            </div>

            <div className="modal-bottom">
              <span>Image: <b>{result.image.name}</b></span>
              <button type="button" onClick={() => setShowMetrics(false)}>Close</button>
            </div>
          </div>
        </div>
      )}

      {preview && (
        <div className="modal-backdrop image-modal-backdrop" onClick={() => setPreview(null)}>
          <div className="image-preview-modal" onClick={(event) => event.stopPropagation()}>
            <div className="image-preview-header">
              <strong>{preview.title}</strong>
              <button type="button" onClick={() => setPreview(null)}>×</button>
            </div>
            <div className="image-preview-body">
              <img src={preview.src} alt={preview.title} />
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

export default App;
