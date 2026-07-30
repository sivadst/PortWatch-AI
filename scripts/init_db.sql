-- Ports reference table
CREATE TABLE IF NOT EXISTS ports (
    port_id VARCHAR(10) PRIMARY KEY,
    port_name VARCHAR(255) NOT NULL,
    country_code CHAR(2) NOT NULL,
    country_code_iso3 CHAR(3),
    unlocode VARCHAR(5),
    latitude DECIMAL(10, 6),
    longitude DECIMAL(10, 6),
    region VARCHAR(50),
    income_group VARCHAR(50),
    is_systemic BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Port activity time series
CREATE TABLE IF NOT EXISTS port_activity (
    id SERIAL PRIMARY KEY,
    port_id VARCHAR(10) REFERENCES ports(port_id),
    date DATE NOT NULL,
    daily_port_calls INT,
    incoming_volume_mt DECIMAL(15, 2),
    outgoing_volume_mt DECIMAL(15, 2),
    congestion_index DECIMAL(8, 4),
    congestion_level VARCHAR(20),
    UNIQUE(port_id, date)
);
