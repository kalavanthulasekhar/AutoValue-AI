
import { useEffect, useState } from "react";

const initialForm = {
  name: "Maruti Swift Dzire VDI",
  year: 2017,
  km_driven: 60000,
  fuel: "Diesel",
  seller_type: "Individual",
  transmission: "Manual",
  owner: "First Owner",
};

const currentYear = new Date().getFullYear();
const HISTORY_KEY = "autovalue_prediction_history";
const API_BASE_URL = (import.meta.env.VITE_API_URL || "").replace(/\/+$/, "");
const API_PREFIX = API_BASE_URL ? "" : "/api";
const apiUrl = (endpoint) => `${API_BASE_URL}${API_PREFIX}/${endpoint}`;

const formatINR = (value) =>
  new Intl.NumberFormat("en-IN", {
    style: "currency",
    currency: "INR",
    maximumFractionDigits: 0,
  }).format(value);

function Field({ label, children }) {
  return (
    <label className="field">
      <span>{label}</span>
      {children}
    </label>
  );
}

function ErrorTable({ title, rows }) {
  return (
    <article className="evaluation-panel residual-table-panel">
      <h4>{title}</h4>
      {rows.length === 0 ? (
        <p>No matching errors in this test split.</p>
      ) : (
        <div className="residual-table-wrap">
          <table className="residual-table">
            <thead>
              <tr><th>Vehicle</th><th>Actual</th><th>Predicted</th><th>Residual</th></tr>
            </thead>
            <tbody>
              {rows.map((row, index) => (
                <tr key={`${row.name}-${row.year}-${index}`}>
                  <th>{row.name}<small>{row.year} · {Number(row.km_driven).toLocaleString("en-IN")} km</small></th>
                  <td>{formatINR(row.actual)}</td>
                  <td>{formatINR(row.predicted)}</td>
                  <td className={row.residual > 0 ? "residual-under" : "residual-over"}>
                    {row.residual > 0 ? "Under" : "Over"} by {formatINR(Math.abs(row.residual))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}
    </article>
  );
}

export default function App() {
  const [form, setForm] = useState(initialForm);
  const [result, setResult] = useState(null);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [selectedCompareIds, setSelectedCompareIds] = useState([]);
  const [showComparison, setShowComparison] = useState(false);
  const [evaluation, setEvaluation] = useState(null);
  const [evaluationLoading, setEvaluationLoading] = useState(true);
  const [evaluationError, setEvaluationError] = useState("");
  const [scatterRange, setScatterRange] = useState("full");
  const [history, setHistory] = useState(() => {
    try {
      const saved = JSON.parse(localStorage.getItem(HISTORY_KEY) || "[]");
      return Array.isArray(saved) ? saved : [];
    } catch {
      return [];
    }
  });
  const [historyFilter, setHistoryFilter] = useState("");

  useEffect(() => {
    fetch(apiUrl("evaluation"))
      .then(async (response) => {
        const data = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(data.error || "Evaluation unavailable.");
        setEvaluation(data);
      })
      .catch((err) => setEvaluationError(err.message))
      .finally(() => setEvaluationLoading(false));
  }, []);

  useEffect(() => {
    if (showComparison) {
      document.getElementById("comparison")?.scrollIntoView({
        behavior: "smooth",
        block: "start",
      });
    }
  }, [showComparison]);

  function changeField(event) {
    const { name, value } = event.target;
    setForm((old) => ({ ...old, [name]: value }));
    setError("");
  }

  useEffect(() => {
    try {
      localStorage.setItem(HISTORY_KEY, JSON.stringify(history));
    } catch (err) {
      console.error("Could not save prediction history:", err);
    }
  }, [history]);

  function toggleCompare(id) {
    if (selectedCompareIds.includes(id)) {
      setSelectedCompareIds((previous) =>
        previous.filter((itemId) => itemId !== id)
      );
    } else if (selectedCompareIds.length >= 4) {
      setError("You can compare up to 4 cars at a time.");
      return;
    } else {
      setSelectedCompareIds((previous) => [...previous, id]);
      setError("");
    }
    setShowComparison(false);
  }

  function removeFromComparison(id) {
    setSelectedCompareIds((previous) =>
      previous.filter((itemId) => itemId !== id)
    );
    if (selectedCompareIds.length <= 2) setShowComparison(false);
  }

  function compareCars() {
    if (selectedCompareIds.length < 2) {
      setError("Select at least 2 saved cars to compare.");
      return;
    }
    setError("");
    setShowComparison(true);
  }

  const comparisonCars = history.filter((item) =>
    selectedCompareIds.includes(item.id)
  );
  const cheapestCar = comparisonCars.length
    ? comparisonCars.reduce((a, b) =>
        a.predicted_price <= b.predicted_price ? a : b
      )
    : null;
  const highestPriceCar = comparisonCars.length
    ? comparisonCars.reduce((a, b) =>
        a.predicted_price >= b.predicted_price ? a : b
      )
    : null;
  const evaluationModels = evaluation?.models ?? (evaluation?.metrics
    ? {
        [evaluation.model_name]: {
          training: evaluation.metrics.train,
          test: evaluation.metrics.test,
          price_ranges: evaluation.price_range_metrics,
        },
      }
    : {});
  const evaluationDataset = evaluation?.dataset;
  const trainingRows = evaluation?.split?.train_rows ?? evaluationDataset?.train_rows;
  const testRows = evaluation?.split?.test_rows ?? evaluationDataset?.test_rows;
  const datasetRows = evaluationDataset?.rows ?? evaluationDataset?.rows_evaluated;
  const evaluationScale = Math.max(
    evaluationDataset?.price_max ?? 0,
    1,
    ...(evaluation?.actual_vs_predicted ?? []).flatMap((point) => [point.actual, point.predicted])
  );
  const activeEvaluation = evaluationModels[evaluation?.model_name];
  const rangeErrors = activeEvaluation?.price_ranges ?? evaluation?.price_range_metrics ?? [];
  const maeToMedian = evaluation?.models?.[evaluation?.model_name]?.test?.mae && evaluationDataset?.median_price
    ? evaluation.models[evaluation.model_name].test.mae / evaluationDataset.median_price
    : null;

  async function predictPrice(event) {
    event.preventDefault();
    setResult(null);
    setError("");

    const year = Number(form.year);
    const kmDriven = Number(form.km_driven);

    if (!form.name.trim()) {
      setError("Please enter the car model.");
      return;
    }

    if (!Number.isInteger(year) || year < 1992 || year > currentYear) {
      setError(`Year must be between 1992 and ${currentYear}.`);
      return;
    }

    if (!Number.isInteger(kmDriven) || kmDriven < 0) {
      setError("Kilometres must be a non-negative whole number.");
      return;
    }

    setLoading(true);

    try {
      const response = await fetch(apiUrl("predict"), {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          ...form,
          name: form.name.trim(),
          year,
          km_driven: kmDriven,
        }),
      });

      const data = await response.json().catch(() => ({}));

      if (!response.ok) {
        throw new Error(data.error || "Prediction failed.");
      }

      if (
        typeof data.predicted_price !== "number" ||
        !Number.isFinite(data.predicted_price)
      ) {
        throw new Error("The API returned an invalid price.");
      }

      setResult(data);
      savePrediction(data);
    } catch (err) {
      setError(
        err.message ||
          "Cannot connect to Flask. Check that your backend is running."
      );
    } finally {
      setLoading(false);
    }
  }

  function resetForm() {
    setForm(initialForm);
    setResult(null);
    setError("");
  }
  
  function savePrediction(data) {
    const record = {
      id: globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random()}`,
      name: form.name.trim(),
      year: Number(form.year),
      km_driven: Number(form.km_driven),
      fuel: form.fuel,
      transmission: form.transmission,
      seller_type: form.seller_type,
      owner: form.owner,
      predicted_price: data.predicted_price,
      model: data.model || "ML prediction model",
      created_at: new Date().toISOString(),
    };

    setHistory((previous) => [record, ...previous]);
  }

  function deletePrediction(id) {
    setHistory((previous) => previous.filter((item) => item.id !== id));
    setSelectedCompareIds((previous) =>
      previous.filter((selectedId) => selectedId !== id)
    );
    setShowComparison(false);
  }

  function clearHistory() {
    if (history.length === 0) return false;

    const confirmed = window.confirm(
      "Are you sure you want to delete all prediction history?"
    );
    if (!confirmed) return false;

    setHistory([]);
    setSelectedCompareIds([]);
    setShowComparison(false);
    return true;
  }

  function reusePrediction(item) {
    setForm({
      name: item.name,
      year: item.year,
      km_driven: item.km_driven,
      fuel: item.fuel,
      transmission: item.transmission,
      seller_type: item.seller_type,
      owner: item.owner,
    });
    setResult(null);
    setError("");
    document.getElementById("valuation")?.scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }

  return (
    <div className="app">
      
<header className="navbar premium-navbar">
  <a className="brand" href="#" aria-label="AutoValue AI home">
    <span className="brand-symbol">A</span>
    <span>
      AutoValue <b>AI</b>
      <small>SMART CAR VALUATION</small>
    </span>
  </a>

  <nav className="premium-nav-links" aria-label="Main navigation">
    <a href="#valuation">Valuation</a>
    <a href="#history">History</a>
    <a href="#evaluation">Model insights</a>
  </nav>

  <div className="premium-nav-actions">
    <a href="#valuation" className="contact-link">Estimate a car <span aria-hidden="true">↗</span></a>
  </div>
</header>


      <main>
        <section className="hero">
          <div className="hero-copy">
            <span className="eyebrow">✦ &nbsp; AI-POWERED CAR VALUATION</span>
            <h1>
              Know what your car
              <br />
              <span>is really worth.</span>
            </h1>
            <p>
              Get an instant estimated selling price using machine learning.
              Enter your vehicle details and let the model do the work.
            </p>
            <div className="features">
              <span>✓ ML-powered estimate</span>
              <span>✓ Instant results</span>
              <span>✓ INR pricing</span>
              <a className="hero-cta" href="#valuation">
                 Get Started <span>↗</span>
              </a>
            </div>
          </div>

          
<div className="car-art">
  <div className="hero-image-glow" />

  <div className="art-label">NEXT-GENERATION MOBILITY</div>
  <div className="art-title">Intelligence meets performance.</div>

  <img
    className="hero-car-image"
    src="/car-hero.webp"
    alt="Silver futuristic sports car with blue lighting"
  />

  <div className="hero-feature feature-one">
    <span>01</span>
    <div>
      <strong>AI Price Estimation</strong>
      <small>Machine learning powered</small>
    </div>
  </div>

  <div className="hero-feature feature-two">
    <span>02</span>
    <div>
      <strong>Smart Valuation</strong>
      <small>Instant vehicle insights</small>
    </div>
  </div>

  <div className="art-foot">
    <span />
    POWERED BY MACHINE LEARNING
  </div>
</div>

        </section>

        
<section className="stats-strip" id="features">
  <div className="stats-intro">
    <span className="stats-icon">✦</span>
    <div>
      <strong>Smarter car valuation</strong>
      <small>Powered by your trained ML pipeline</small>
    </div>
  </div>

  <div className="stat-item">
    <strong>01</strong>
    <span>Enter car details</span>
  </div>

  <div className="stat-item">
    <strong>02</strong>
    <span>AI price estimate</span>
  </div>

  <div className="stat-item">
    <strong>03</strong>
    <span>Instant result</span>
  </div>

  <a className="strip-link" href="#valuation">
    Estimate a car ↗
  </a>
</section>


        <section className="workspace" id="valuation">
          <div className="section-heading">
            <div>
              <span className="eyebrow">YOUR VEHICLE</span>
              <h2>Tell us about your car</h2>
              <p>Complete the details to calculate an estimated selling price.</p>
            </div>
            <span className="required">* Required fields</span>
          </div>

          <div className="dashboard-grid">
            <form className="form-card" onSubmit={predictPrice}>
              <div className="card-heading">
                <span className="number">01</span>
                <div>
                  <h3>Vehicle information</h3>
                  <p>Enter your car specifications</p>
                </div>
              </div>

              <div className="form-grid">
                <Field label="Car name / model *">
                  <input
                    name="name"
                    value={form.name}
                    onChange={changeField}
                    placeholder="e.g. Maruti Swift Dzire VDI"
                    required
                  />
                </Field>

                <Field label="Manufacturing year *">
                  <input
                    name="year"
                    type="number"
                    min="1992"
                    max={currentYear}
                    step="1"
                    value={form.year}
                    onChange={changeField}
                    required
                  />
                </Field>

                <Field label="Kilometres driven *">
                  <input
                    name="km_driven"
                    type="number"
                    min="0"
                    step="1"
                    value={form.km_driven}
                    onChange={changeField}
                    required
                  />
                </Field>

                <Field label="Fuel type *">
                  <select name="fuel" value={form.fuel} onChange={changeField}>
                    {["Diesel", "Petrol", "CNG", "LPG", "Electric"].map((x) => (
                      <option key={x}>{x}</option>
                    ))}
                  </select>
                </Field>

                <Field label="Transmission *">
                  <select
                    name="transmission"
                    value={form.transmission}
                    onChange={changeField}
                  >
                    <option>Manual</option>
                    <option>Automatic</option>
                  </select>
                </Field>

                <Field label="Seller type *">
                  <select
                    name="seller_type"
                    value={form.seller_type}
                    onChange={changeField}
                  >
                    <option>Individual</option>
                    <option>Dealer</option>
                    <option>Trustmark Dealer</option>
                  </select>
                </Field>

                <Field label="Ownership history *">
                  <select
                    name="owner"
                    value={form.owner}
                    onChange={changeField}
                  >
                    <option>First Owner</option>
                    <option>Second Owner</option>
                    <option>Third Owner</option>
                    <option>Fourth &amp; Above Owner</option>
                    <option>Test Drive Car</option>
                  </select>
                </Field>
              </div>

              {error && <div className="error" role="alert">⚠ {error}</div>}

              <div className="form-actions">
                <button type="button" className="reset-btn" onClick={resetForm}>
                  Reset
                </button>
                
                  <button className="predict-btn" disabled={loading} type="submit">
                    {loading ? (
                       <>
                          <span className="spinner" aria-hidden="true" />
                              Predicting...
                          </>
                     ) : (
                       <>
                          Predict car price <span aria-hidden="true">↗</span>
                        </>
                      )}
                  </button>
               </div>
              <p className="form-note">
                Your estimate is generated by your trained ML pipeline.
              </p>
            </form>

            <aside
              className="result-card"
              aria-live="polite"
              aria-atomic="true"
              >
              <div className="result-heading">
                <span className="rupee">₹</span>
                <svg className="result-car-icon" viewBox="0 0 24 16" aria-hidden="true">
                  <path
                    d="M2 10.5 4 6h3l2.2-3h5.6l3.2 3 3 1.5 1 3v2H2z"
                    fill="none"
                    stroke="currentColor"
                    strokeLinecap="round"
                    strokeLinejoin="round"
                    strokeWidth="1.5"
                  />
                  <circle cx="6" cy="12.5" r="1.5" fill="#122139" stroke="currentColor" strokeWidth="1.5" />
                  <circle cx="18" cy="12.5" r="1.5" fill="#122139" stroke="currentColor" strokeWidth="1.5" />
                </svg>
                <span>PRICE ESTIMATION</span>
              </div>

              {result ? (
                <div className="result-body">
                  <span className="result-label">YOUR ESTIMATED VALUE</span>
                  <div className="price">{formatINR(result.predicted_price)}</div>
                  <p className="result-description">
                    Estimated selling price based on your vehicle details.
                  </p>
                  <div className="result-divider" />
                  <div className="result-row">
                    <span>Vehicle</span><strong>{form.name}</strong>
                  </div>
                  <div className="result-row">
                    <span>Year</span><strong>{form.year}</strong>
                  </div>
                  <div className="result-row">
                    <span>Distance</span>
                    <strong>{Number(form.km_driven).toLocaleString("en-IN")} km</strong>
                  </div>
                  <div className="model-tag">
                    <span className="dot" />
                    {result.model || "ML prediction model"}
                  </div>
                  <p className="disclaimer">
                    {result.note || "Estimate only; actual selling prices may vary."}
                  </p>
                  <button
                    className="reset-btn full-width"
                    onClick={() => setResult(null)}
                    type="button"
                  >
                    Estimate another car
                  </button>
                </div>
              ) : (
                <div className="empty-result">
                  <div className="empty-visual">
                    <div className="empty-orbit" />
                    <span className="sparkle">✦</span>
                    <svg viewBox="0 0 220 100" aria-hidden="true">
                      <path
                        d="M15 68 L30 47 L70 39 L94 17 Q100 11 112 12 L145 15 Q155 17 166 34 L181 44 L200 52 L205 69 L15 69Z"
                        fill="#23466c"
                        stroke="#8cc5ff"
                        strokeWidth="2"
                      />
                      <path d="M79 37 L101 17 L113 17 L113 38Z" fill="#91c7f6" />
                      <path d="M121 17 L145 19 Q153 21 161 38 L121 38Z" fill="#91c7f6" />
                      <circle cx="59" cy="70" r="14" fill="#091221" stroke="#9ab4d3" strokeWidth="3" />
                      <circle cx="166" cy="70" r="14" fill="#091221" stroke="#9ab4d3" strokeWidth="3" />
                    </svg>
                  </div>
                  <h3>Your estimate appears here</h3>
                  <p>
                    Enter your vehicle details and run a prediction to see its
                    estimated selling price.
                  </p>
                  <div className="steps">
                    <span><b>1</b> Enter details</span>
                    <span><b>2</b> Predict price</span>
                    <span><b>3</b> View estimate</span>
                  </div>
                </div>
              )}
            </aside>
          </div>
        </section>

        
<section className="history-section" id="history">
  <div className="history-heading">
    <div>
      <span className="eyebrow">YOUR ACTIVITY</span>
      <h2>Prediction history</h2>
      <p>Review your previous car valuations and reuse vehicle details.</p>
    </div>
  </div>

    
<div className="history-actions">
  <span className="history-count">
    {history.length} {history.length === 1 ? "record" : "records"}
  </span>

  <button
    type="button"
    className="compare-main-btn"
    onClick={compareCars}
    disabled={selectedCompareIds.length < 2}
  >
    Compare Selected ({selectedCompareIds.length})
  </button>
  {history.length > 0 && (
    <span className="compare-hint" aria-live="polite">
      {selectedCompareIds.length < 2
        ? "Select 2–4 cars to compare"
        : `${selectedCompareIds.length} cars selected`}
    </span>
  )}

  <button
    type="button"
    className="history-clear-btn"
    onClick={clearHistory}
    disabled={history.length === 0}
  >
    Clear history
  </button>
</div>


  <div className="history-toolbar">
    <input
      type="search"
      aria-label="Search prediction history"
      placeholder="Search by car model..."
      value={historyFilter}
      onChange={(event) => setHistoryFilter(event.target.value)}
    />

    <span>Newest predictions first</span>
  </div>

  {history.length === 0 ? (
    <div className="history-empty">
      <div className="history-empty-icon">↗</div>
      <h3>No predictions yet</h3>
      <p>
        Your successful price estimates will appear here after you predict
        your first car.
      </p>
      <a href="#valuation" className="history-primary-link">
        Estimate your first car ↗
      </a>
    </div>
  ) : (
    <div className="history-list">
      {history
        .filter((item) =>
          item.name.toLowerCase().includes(historyFilter.trim().toLowerCase())
        )
        .map((item) => (
          <article className="history-item" key={item.id}>
            <div className="history-car-icon">🚘</div>

            <div className="history-car-details">
              <h3>{item.name}</h3>
              <p>
                {item.year} · {item.fuel} · {item.transmission}
              </p>
              <p>
                {Number(item.km_driven).toLocaleString("en-IN")} km ·{" "}
                {item.owner}
              </p>
              <time dateTime={item.created_at}>
                {new Date(item.created_at).toLocaleString("en-IN", {
                  dateStyle: "medium",
                  timeStyle: "short",
                })}
              </time>
            </div>

            <div className="history-price">
              <span>Estimated price</span>
              <strong>{formatINR(item.predicted_price)}</strong>
              <small>{item.model}</small>
            </div>

            <div className="history-item-actions">
  <label className="compare-select">
    <input
      type="checkbox"
      checked={selectedCompareIds.includes(item.id)}
      onChange={() => toggleCompare(item.id)}
      aria-label={`Select ${item.name} for comparison`}
    />
    <span>Compare</span>
  </label>

  <button
    type="button"
    onClick={() => reusePrediction(item)}
    aria-label={`Reuse details for ${item.name}`}
  >
    Reuse
  </button>

  <button
    type="button"
    className="history-delete-btn"
    onClick={() => deletePrediction(item.id)}
    aria-label={`Delete ${item.name} prediction`}
  >
    Delete
  </button>
            </div>
          </article>
        ))}

      {history.filter((item) =>
        item.name.toLowerCase().includes(historyFilter.trim().toLowerCase())
      ).length === 0 && (
        <div className="history-empty history-no-results">
          <h3>No matching cars</h3>
          <p>Try a different car model name.</p>
        </div>
      )}
    </div>
  )}

  
{showComparison && comparisonCars.length >= 2 && (
  <section className="comparison-section" id="comparison">
    <div className="comparison-heading">
      <div>
        <span className="eyebrow">SIDE-BY-SIDE ANALYSIS</span>
        <h2>Compare your cars</h2>
        <p>
          Compare estimated prices and vehicle specifications using your
          saved predictions.
        </p>
      </div>

      <button
        type="button"
        className="history-clear-btn"
        onClick={() => setShowComparison(false)}
      >
        Close comparison
      </button>
    </div>

    <div className="comparison-summary">
      <div className="comparison-summary-card">
        <span>Cars compared</span>
        <strong>{comparisonCars.length}</strong>
        <small>Saved predictions</small>
      </div>

      <div className="comparison-summary-card">
        <span>Lowest estimate</span>
        <strong>{formatINR(cheapestCar.predicted_price)}</strong>
        <small>{cheapestCar.name}</small>
      </div>

      <div className="comparison-summary-card">
        <span>Highest estimate</span>
        <strong>{formatINR(highestPriceCar.predicted_price)}</strong>
        <small>{highestPriceCar.name}</small>
      </div>
    </div>

    <div className="comparison-table-wrap">
      <table className="comparison-table">
        <thead>
          <tr>
            <th>Vehicle details</th>
            {comparisonCars.map((car) => (
              <th key={car.id}>
                <div className="comparison-car-heading">
                  <span className="comparison-car-icon">🚘</span>
                  <strong>{car.name}</strong>
                  <button
                    type="button"
                    onClick={() => removeFromComparison(car.id)}
                    aria-label={`Remove ${car.name} from comparison`}
                    title="Remove from comparison"
                  >
                    ×
                  </button>
                </div>
              </th>
            ))}
          </tr>
        </thead>

        <tbody>
          <tr className="comparison-price-row">
            <th>Estimated price</th>
            {comparisonCars.map((car) => (
              <td key={car.id}>
                <strong>{formatINR(car.predicted_price)}</strong>
                {car.id === cheapestCar.id && (
                  <span className="price-badge">Lowest estimate</span>
                )}
                {car.id === highestPriceCar.id &&
                  highestPriceCar.id !== cheapestCar.id && (
                    <span className="price-badge high-badge">
                      Highest estimate
                    </span>
                  )}
              </td>
            ))}
          </tr>

          <tr>
            <th>Manufacturing year</th>
            {comparisonCars.map((car) => (
              <td key={car.id}>{car.year}</td>
            ))}
          </tr>

          <tr>
            <th>Kilometres driven</th>
            {comparisonCars.map((car) => (
              <td key={car.id}>
                {Number(car.km_driven).toLocaleString("en-IN")} km
              </td>
            ))}
          </tr>

          <tr>
            <th>Fuel type</th>
            {comparisonCars.map((car) => (
              <td key={car.id}>{car.fuel}</td>
            ))}
          </tr>

          <tr>
            <th>Transmission</th>
            {comparisonCars.map((car) => (
              <td key={car.id}>{car.transmission}</td>
            ))}
          </tr>

          <tr>
            <th>Seller type</th>
            {comparisonCars.map((car) => (
              <td key={car.id}>{car.seller_type}</td>
            ))}
          </tr>

          <tr>
            <th>Ownership</th>
            {comparisonCars.map((car) => (
              <td key={car.id}>{car.owner}</td>
            ))}
          </tr>

          <tr>
            <th>Prediction model</th>
            {comparisonCars.map((car) => (
              <td key={car.id}>{car.model}</td>
            ))}
          </tr>
        </tbody>
      </table>
    </div>

    <div className="comparison-chart">
      <div className="comparison-chart-heading">
        <div>
          <h3>Price comparison</h3>
          <p>Estimated selling prices in Indian rupees</p>
        </div>
      </div>

      {(() => {
        const maxPrice = Math.max(
          ...comparisonCars.map((car) => car.predicted_price),
          1
        );

        return (
          <div className="comparison-bars">
            {comparisonCars.map((car) => (
              <div className="comparison-bar-item" key={car.id}>
                <div className="comparison-bar-label">
                  <span>{car.name}</span>
                  <strong>{formatINR(car.predicted_price)}</strong>
                </div>

                <div
                  className="comparison-bar-track"
                  role="img"
                  aria-label={`${car.name}: ${formatINR(car.predicted_price)}`}
                >
                  <div
                    className="comparison-bar-fill"
                    style={{
                      width: `${Math.max(
                        2,
                        (car.predicted_price / maxPrice) * 100
                      )}%`,
                    }}
                  />
                </div>
              </div>
            ))}
          </div>
        );
      })()}
    </div>

    <p className="comparison-disclaimer">
      These figures are historical model estimates saved in your browser.
      They are not guaranteed market prices. Differences may reflect vehicle
      specifications and model prediction error.
    </p>

    <div className="comparison-footer">
      <button
        type="button"
        className="reset-btn"
        onClick={() => setShowComparison(false)}
      >
        Close comparison
      </button>

      <button
        type="button"
        className="predict-btn"
        onClick={() => {
          setShowComparison(false);
          document.getElementById("history")?.scrollIntoView({
            behavior: "smooth",
            block: "start",
          });
        }}
      >
        Choose different cars ↗
      </button>
    </div>
  </section>
)}

</section>

<section className="evaluation-section" id="evaluation">
  <div className="history-heading">
    <div>
      <span className="eyebrow">MODEL QUALITY</span>
      <h2>Model evaluation</h2>
      <p>Candidate models share the same fixed holdout split. Training scores are identified separately.</p>
    </div>
  </div>
  {evaluation ? (
    <>
      <div className="evaluation-meta">
        Active pipeline: <strong>{evaluation.production_model_name || evaluation.model_name}</strong> · {trainingRows.toLocaleString()} training rows · {testRows.toLocaleString()} test rows
        {evaluation.candidate_winner && <span> · Lowest holdout MAE: {evaluation.candidate_winner}</span>}
      </div>
      {evaluation.selection_note && <p className="evaluation-caveat">{evaluation.selection_note}</p>}
      {evaluation.candidate_winner && !evaluation.model_updated && (
        <p className="evaluation-caveat">{evaluation.candidate_winner} had the lowest MAE, but did not meet the replacement threshold. {evaluation.model_name} remains the production model.</p>
      )}
      <div className="evaluation-models">
        {Object.entries(evaluationModels).map(([name, scores]) => (
          <article className="evaluation-card" key={name}>
            <h3>{name}</h3>
            <h4>Holdout test set</h4>
            <p>MAE <strong>{formatINR(scores.test.mae)}</strong></p>
            {scores.test.rmse != null && <p>RMSE <strong>{formatINR(scores.test.rmse)}</strong></p>}
            <p>R² <strong>{scores.test.r2.toFixed(3)}</strong></p>
            <details>
              <summary>Training metrics</summary>
              <p>MAE {formatINR(scores.training.mae)} · RMSE {formatINR(scores.training.rmse)} · R² {scores.training.r2.toFixed(3)}</p>
            </details>
          </article>
        ))}
      </div>
      <div className="metric-guide">
        <p><strong>MAE</strong> is the average absolute prediction error. The active model’s test MAE is {formatINR(activeEvaluation.test.mae)}{maeToMedian ? `, about ${(maeToMedian * 100).toFixed(0)}% of the dataset median price` : ""}.</p>
        <p><strong>RMSE</strong> measures error too, but penalizes large misses more strongly.</p>
        <p><strong>R²</strong> means the model explains about {(activeEvaluation.test.r2 * 100).toFixed(1)}% of price variation on this test set. {activeEvaluation.test.r2 < 0.5 ? "This score indicates substantial unexplained variation and prediction error." : "Higher values indicate more variation explained."}</p>
      </div>
      <div className="evaluation-columns">
        <article className="evaluation-panel">
          <h3>Actual vs predicted · {scatterRange === "common" ? "common price range" : "full range"}</h3>
          <div className="scatter-controls" role="group" aria-label="Scatter plot price range">
            <button type="button" aria-pressed={scatterRange === "full"} onClick={() => setScatterRange("full")}>Full range</button>
            <button type="button" aria-pressed={scatterRange === "common"} onClick={() => setScatterRange("common")}>Zoom under ₹10 lakh</button>
          </div>
          <div className="scatter-frame">
            <span className="scatter-y-label">Predicted price (INR)</span>
            <div className="evaluation-scatter" role="img" aria-label={`Actual versus predicted prices, ${scatterRange === "common" ? "zoomed to actual prices under ten lakh rupees" : "full price range"}`}>
            {evaluation.actual_vs_predicted
              .filter((point) => scatterRange !== "common" || point.actual <= 1_000_000)
              .map((point, index) => {
                const max = scatterRange === "common" ? 1_000_000 : evaluationScale;
                return <span key={`${point.actual}-${index}`} style={{ left: `${Math.min(97, Math.max(2, point.actual / max * 100))}%`, bottom: `${Math.min(97, Math.max(2, point.predicted / max * 100))}%` }} title={`Actual ${formatINR(point.actual)}, predicted ${formatINR(point.predicted)}`} />;
              })}
            </div>
            <span className="scatter-x-label">Actual price (INR)</span>
          </div>
          <small>Each dot is a held-out vehicle. Actual price increases left to right; predicted price increases bottom to top. Both are linear scales from ₹0 to {formatINR(scatterRange === "common" ? 1_000_000 : evaluationScale)}. The perfect-estimate diagonal runs bottom-left to top-right. Zoom includes actual prices up to ₹10 lakh; predictions above that limit are clipped.</small>
          <p className="residual-note">Residual (actual − predicted): mean {formatINR(evaluation.residual_diagnostics.mean_actual_minus_predicted)}, median {formatINR(evaluation.residual_diagnostics.median_actual_minus_predicted)}; {Math.round(evaluation.residual_diagnostics.underprediction_rate * 100)}% of test predictions are below actual prices. The mean can be shifted by a few large misses, so compare the price-band errors too.</p>
        </article>
        <article className="evaluation-panel">
          <h3>Top feature importance</h3>
          {evaluation.feature_importance?.length ? evaluation.feature_importance.map((feature) => <div className="importance-row" key={feature.feature}>
            <span>{feature.feature}</span><b>{(feature.importance * 100).toFixed(1)}%</b>
            <i style={{ width: `${Math.max(1, feature.importance / evaluation.feature_importance[0].importance * 100)}%` }} />
          </div>) : <p>Feature importance was not included in this evaluation run.</p>}
        </article>
      </div>
      <section className="residual-analysis">
        <div className="evaluation-panel residual-overview">
          <h3>Residual inspection · production model</h3>
          <p>Residual is actual minus predicted: positive values are underpredictions; negative values are overpredictions. Rows are from the held-out test split.</p>
        </div>
        <div className="residual-table-grid">
          <ErrorTable title="Largest overpredictions" rows={evaluation.largest_overpredictions ?? []} />
          <ErrorTable title="Largest underpredictions" rows={evaluation.largest_underpredictions ?? []} />
        </div>
        <ErrorTable title="Largest errors above ₹10 lakh actual price" rows={evaluation.largest_high_price_errors ?? []} />
      </section>
      <article className="evaluation-panel price-range-panel">
        <h3>Test MAE by actual price range · {evaluation.production_model_name || evaluation.model_name}</h3>
        <p>Lower MAE means smaller average absolute errors within that price band.</p>
        {(() => {
          const maxMae = Math.max(1, ...rangeErrors.map((range) => range.mae));
          return <div className="price-range-chart">
            {rangeErrors.map((range) => (
              <div className="price-range-row" key={range.range}>
                <div className="price-range-label">
                  <span>{range.range} <small>({range.count} cars)</small></span>
                  <strong>{formatINR(range.mae)}</strong>
                </div>
                <small className="price-range-bias">
                  Mean residual {range.mean_actual_minus_predicted > 0 ? "underprediction" : "overprediction"}: {formatINR(Math.abs(range.mean_actual_minus_predicted))} · {Math.round(range.underprediction_rate * 100)}% underpredicted
                </small>
                <div className="price-range-track" role="img" aria-label={`${range.range}: MAE ${formatINR(range.mae)}`}>
                  <i style={{ width: `${Math.max(1, range.mae / maxMae * 100)}%` }} />
                </div>
              </div>
            ))}
          </div>;
        })()}
      </article>
      <div className="dataset-insights">
        <h3>Dataset snapshot</h3>
        <span>{datasetRows.toLocaleString()} vehicles evaluated</span>
        {evaluationDataset.median_price != null && <span>Median price {formatINR(evaluationDataset.median_price)}</span>}
        {evaluationDataset.price_min != null && evaluationDataset.price_max != null && <span>Range {formatINR(evaluationDataset.price_min)}–{formatINR(evaluationDataset.price_max)}</span>}
        {evaluationDataset.missing_values != null && <span>{evaluationDataset.missing_values} missing values in model fields</span>}
        {evaluationDataset.top_fuels && <span>Fuel mix {Object.entries(evaluationDataset.top_fuels).map(([fuel, count]) => `${fuel} ${count}`).join(" · ")}</span>}
      </div>
      <section className="dataset-insights variant-insights">
        <h3>Car name and variant coverage · training rows</h3>
        <span>{evaluationDataset.unique_names_training.toLocaleString()} unique names</span>
        <span>{evaluationDataset.names_seen_once_training.toLocaleString()} names appear once</span>
        <span>{evaluationDataset.high_price_rows_training.toLocaleString()} cars above ₹10 lakh</span>
        <span>{evaluationDataset.high_price_unique_names_training.toLocaleString()} unique names above ₹10 lakh</span>
        <span>{evaluationDataset.high_price_names_seen_at_most_twice_training.toLocaleString()} high-price names have only 1–2 training examples</span>
        <div className="variant-example-list">
          <h4>Rare high-price variants (1–2 training examples)</h4>
          {evaluationDataset.rare_high_price_name_examples.length ? evaluationDataset.rare_high_price_name_examples.map((item) => (
            <span key={item.name}>{item.name} · {item.training_count} examples · median {formatINR(item.median_price)}</span>
          )) : <span>No high-price variants fall below the frequency threshold.</span>}
          <h4>Other names with at most two training examples</h4>
          {evaluationDataset.rare_name_examples.map((item) => (
            <span key={item.name}>{item.name} · {item.training_count} example{item.training_count === 1 ? "" : "s"}</span>
          ))}
        </div>
      </section>
    </>
  ) : (
    <div className="history-empty" role={evaluationLoading ? "status" : "alert"}>
      <h3>{evaluationLoading ? "Loading model insights" : "Evaluation data unavailable"}</h3>
      <p>{evaluationLoading ? "Fetching evaluation metrics for the trained model…" : evaluationError}</p>
    </div>
  )}
</section>


        <div className="info-banner">
          <span className="info-icon">i</span>
          <p>
            <strong>About this estimate</strong>
            The result is based on historical training data. Actual resale prices
            vary with condition, location, service history, and market demand.
            Rare car variants may be less accurately estimated.
          </p>
        </div>
      </main>

      <footer className="footer">
        <strong>AutoValue AI</strong>
        <span>Machine learning for smarter car valuation.</span>
        <span>© {currentYear} · Car Price Predictor</span>
      </footer>
    </div>
  );
}
