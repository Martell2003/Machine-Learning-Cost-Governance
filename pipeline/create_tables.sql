-- ── PROJECTS ────────────────────────────────────────────────────────
CREATE TABLE projects (
    project_id        SERIAL PRIMARY KEY,
    project_name      TEXT NOT NULL,
    budget_at_completion  NUMERIC(12,2),     -- BAC: total planned budget
    total_periods     INTEGER,               -- total reporting periods
    train_test_flag   VARCHAR(5),            -- 'train' or 'test'
    project_size      VARCHAR(10),           -- 'small', 'medium', 'large'
    created_at        TIMESTAMP DEFAULT NOW()
);

-- ── PERIODS ─────────────────────────────────────────────────────────
CREATE TABLE periods (
    period_id         SERIAL PRIMARY KEY,
    project_id        INTEGER REFERENCES projects(project_id) ON DELETE CASCADE,
    period_number     INTEGER NOT NULL,
    planned_value     NUMERIC(12,2),         -- PV
    earned_value      NUMERIC(12,2),         -- EV
    actual_cost       NUMERIC(12,2),         -- AC
    cumulative_pv     NUMERIC(12,2),
    cumulative_ev     NUMERIC(12,2),
    cumulative_ac     NUMERIC(12,2),
    UNIQUE(project_id, period_number)
);

-- ── FEATURES ────────────────────────────────────────────────────────
CREATE TABLE features (
    feature_id        SERIAL PRIMARY KEY,
    project_id        INTEGER REFERENCES projects(project_id) ON DELETE CASCADE,
    prediction_period INTEGER,               -- period at which features were calculated
    cpi_trend         NUMERIC(8,4),          -- EV / AC at 20% mark
    sv_trajectory     NUMERIC(12,2),         -- EV - PV at 20% mark
    burn_rate_ratio   NUMERIC(8,4),          -- AC / BAC at 20% mark
    schedule_pressure NUMERIC(8,4),          -- SV / BAC at 20% mark
    actual_final_variance NUMERIC(8,4),      -- known outcome — target for training
    created_at        TIMESTAMP DEFAULT NOW(),
    UNIQUE(project_id)
);

-- ── FORECASTS ───────────────────────────────────────────────────────
CREATE TABLE forecasts (
    forecast_id       SERIAL PRIMARY KEY,
    project_id        INTEGER REFERENCES projects(project_id) ON DELETE CASCADE,
    feature_id        INTEGER REFERENCES features(feature_id),  -- Link to features used
    forecast_period   INTEGER,               -- period at which forecast was generated
    rf_variance_pct   NUMERIC(8,4),          -- Random Forest predicted variance %
    arima_variance_pct NUMERIC(8,4),         -- ARIMA predicted variance %
    ensemble_variance_pct NUMERIC(8,4),      -- Combined forecast
    rf_confidence_low  NUMERIC(8,4),         -- RF lower confidence bound
    rf_confidence_high NUMERIC(8,4),         -- RF upper confidence bound
    generated_at      TIMESTAMP DEFAULT NOW()
);

-- ── SHAP_VALUES ─────────────────────────────────────────────────────
CREATE TABLE shap_values (
    shap_id           SERIAL PRIMARY KEY,
    forecast_id       INTEGER REFERENCES forecasts(forecast_id) ON DELETE CASCADE,
    feature_name      TEXT NOT NULL,         -- 'cpi_trend', 'sv_trajectory', etc.
    shap_value        NUMERIC(10,6),         -- how much this feature moved the forecast
    impact_direction  VARCHAR(4),            -- 'up' or 'down' — renamed for clarity
    base_value        NUMERIC(8,4)           -- model average prediction (starting point)
);

-- ── ALERTS ──────────────────────────────────────────────────────────
CREATE TABLE alerts (
    alert_id          SERIAL PRIMARY KEY,
    forecast_id       INTEGER REFERENCES forecasts(forecast_id) ON DELETE CASCADE,
    project_id        INTEGER REFERENCES projects(project_id) ON DELETE CASCADE,
    tier              INTEGER NOT NULL,      -- 1=Green, 2=Amber, 3=Red, 4=Critical
    status            VARCHAR(10) NOT NULL,  -- 'GREEN', 'AMBER', 'RED', 'CRITICAL'
    prescribed_action TEXT,
    responsible_role  TEXT,
    response_window   TEXT,                  -- 'Within 14 days' etc.
    acknowledged      BOOLEAN DEFAULT FALSE,
    triggered_at      TIMESTAMP DEFAULT NOW()
);

-- ── VALIDATION_RESULTS ──────────────────────────────────────────────
CREATE TABLE validation_results (
    result_id         SERIAL PRIMARY KEY,
    validator_code    TEXT,                  -- 'V1', 'V2', 'V3' — anonymised
    evaluation_date   DATE,
    actionability_score   INTEGER CHECK (actionability_score BETWEEN 1 AND 5),
    explainability_score  INTEGER CHECK (explainability_score BETWEEN 1 AND 5),
    integrability_score   INTEGER CHECK (integrability_score BETWEEN 1 AND 5),
    overall_score     NUMERIC(4,2),         -- average of the three
    comments          TEXT,
    recommendation    VARCHAR(20)           -- 'approved', 'revision', 'rejected'
);

-- ── INDEXES ─────────────────────────────────────────────────────────
CREATE INDEX idx_periods_project ON periods(project_id);
CREATE INDEX idx_periods_period ON periods(project_id, period_number);
CREATE INDEX idx_features_project ON features(project_id);
CREATE INDEX idx_forecasts_project ON forecasts(project_id);
CREATE INDEX idx_forecasts_feature ON forecasts(feature_id);
CREATE INDEX idx_shap_forecast ON shap_values(forecast_id);
CREATE INDEX idx_alerts_forecast ON alerts(forecast_id);
CREATE INDEX idx_alerts_tier ON alerts(tier);
CREATE INDEX idx_alerts_project ON alerts(project_id);