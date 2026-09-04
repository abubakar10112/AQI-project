-- =============================================================================
-- Supabase table for AQI feature storage
-- Run this in your Supabase SQL Editor (https://supabase.com/dashboard)
-- =============================================================================

CREATE TABLE IF NOT EXISTS aqi_features (
    timestamp        TIMESTAMPTZ PRIMARY KEY,

    -- Target
    us_aqi           FLOAT8 NOT NULL,

    -- Weather features
    temperature_2m       FLOAT8,
    relative_humidity_2m FLOAT8,
    surface_pressure     FLOAT8,
    precipitation        FLOAT8,
    cloud_cover          FLOAT8,
    wind_speed_10m       FLOAT8,
    wind_direction_10m   FLOAT8,
    wind_gusts_10m       FLOAT8,

    -- Pollutant features
    pm2_5            FLOAT8,
    pm10             FLOAT8,
    nitrogen_dioxide FLOAT8,
    sulphur_dioxide  FLOAT8,
    ozone            FLOAT8,
    carbon_monoxide  FLOAT8,

    -- Time-based features
    hour             FLOAT8,
    day_of_week      FLOAT8,
    day_of_month     FLOAT8,
    month            FLOAT8,
    is_weekend       FLOAT8,
    season           FLOAT8,

    -- Lahore-specific features
    is_smog_season        FLOAT8,
    is_crop_burning_season FLOAT8,
    is_brick_kiln_active  FLOAT8,
    wind_from_east        FLOAT8,
    days_since_rain       FLOAT8,

    -- Derived / lag features
    aqi_change_rate       FLOAT8,
    aqi_rolling_mean_24h  FLOAT8,
    aqi_rolling_std_24h   FLOAT8,
    aqi_lag_1h            FLOAT8,
    aqi_lag_3h            FLOAT8,
    aqi_lag_6h            FLOAT8,
    aqi_lag_12h           FLOAT8,
    aqi_lag_24h           FLOAT8,
    temp_x_humidity       FLOAT8,
    wind_x_pm25           FLOAT8
);

-- Index for fast range queries used by training and inference
CREATE INDEX IF NOT EXISTS idx_aqi_features_timestamp
    ON aqi_features (timestamp DESC);

-- Enable Row Level Security (allow all for service role key)
ALTER TABLE aqi_features ENABLE ROW LEVEL SECURITY;

CREATE POLICY "Allow all for service role"
    ON aqi_features
    FOR ALL
    USING (true)
    WITH CHECK (true);
