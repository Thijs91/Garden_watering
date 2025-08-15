# Garden watering
Determine the needed amount of water to water the garden based on FAO56 method with a local weather station.

## ET-Based Garden Irrigation (FAO-56 Method)

This Home Assistant configuration implements **ET-based (evapotranspiration) irrigation control** for a garden, based on the **FAO-56 guideline** for computing reference evapotranspiration. It calculates how much moisture the soil loses each day due to weather (evaporation + plant transpiration) and then adjusts watering time to replace only what’s needed.

### What it does
- **Computes daily ET** from weather inputs (temperature, humidity, wind speed, and sunlight/solar proxy where available) using FAO-56 reference ET equations.
- **Tracks a water balance (“bucket”)** by subtracting evapotranspiration and scales for seasons/crop coefficients.
- **Derives per-zone run time** so irrigation compensates for net moisture loss, with configurable minimums and maximums.
- **Gives values to integrate with switches/valves** exposed in Home Assistant (relays, smart plugs, etc.).

### Why it’s useful
- Waters precisely what the plants used, **saving water** and preventing over-watering.
- Adapts automatically to changing weather without manual re-tuning of timers.
- Works with existing zones, sensors, and automations.

### Inputs & assumptions
- Weather entities: temperature, humidity, wind speed, solar radiation or illuminance proxy.
- Rainfall source: sensor or weather forecast entity.
- Optional coefficients for crop type, soil type, and seasonal adjustments.

### Outputs
- Sensors for ET and water balance.
- Calculated **irrigation duration per zone**.

> **Note:** Update the YAML entity IDs, services, and coefficients to match your Home Assistant setup.

### Installation:
Add "irrigation_et.yaml" to /config/packages/

Add in configuration.yaml
```
homeassistant:
  packages: !include_dir_named packages
```
