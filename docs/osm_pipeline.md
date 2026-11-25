# OSM → transmission-line inputs: pipeline sketch

Goal: turn OpenStreetMap power data into `r2x-txlines` line objects that the thermal,
sag and St. Clair calculations can run on.

Status: stages 1, 2 and 4 built in `transmissionlines.io.osm`: `read_power` and
`assemble_circuits` (need the `osm` extra), `line_from_circuit` and `route_geometry` (don't).
`line_from_circuit(record, configuration)` returns a `RoutedTransmissionLine` when both ends
reach substations and towers are mapped, else a `CrossSectionTransmissionLine` to rate along
`route_geometry(record)`. Each substation gets one bus location, and routes are extended to it,
so lines meeting there share a bus. Not yet: the stage 3 defaults table (the `configuration`
is passed in), elevation, per-phase folding at junctions, `route=power` relations, area
boundaries for identifying ties, the GeoParquet cache.
Delaware (≥69 kV): 523 circuits, all build and register in one system; 122 routed.
Scope per the PI: mainly lines above 220 kV, inter-area and possibly intra-area.

```
 OSM extract ──► 1. extract ──► 2. assemble ──► 3. enrich ──► 4. build ──► calculations
 (.osm.pbf)      raw elements    circuits        + defaults,     Cross-section /
                 w/ node IDs     (towers, ends)   elevation       RoutedTransmissionLine
```

## 1. Data: from where, in what format

| Need | Source | Format | Notes |
|---|---|---|---|
| Routes, towers, substations, tags | **Geofabrik extracts** (`download.geofabrik.de/north-america/us/<state>-latest.osm.pbf`, or multi-state regions) | OSM PBF | Free, refreshed daily, any size from a state to all of North America. Keeps node IDs and way→node order. |
| Same, small areas / ad hoc | Overpass API (what `belize_open_infras_maps` uses) | Overpass JSON | Fine for a county or one line. Keep the **raw JSON**, not GeoJSON: GeoJSON drops the way's node list. Public servers are often overloaded, and big areas time out. |
| Conductor, ground wire, insulators | `r2x-txlines` catalog | existing records | OSM almost never names the conductor. |
| Tower wire positions (x, y) | standard tower-design library (to build) | `TowerConfiguration` per design | OSM has no wire positions. |
| Elevation | USGS 3DEP (or similar DEM) | GeoTIFF / point query | OSM `ele` is filled on 0% of towers. |
| Weather | WTK + NSRDB via `fetch_weather` | already built | Unchanged. |

Measured tag coverage on the US samples in `belize_open_infras_maps` (20,566 `power=line`
ways, 282,243 towers and poles): `voltage` 85%, `cables` 77%, `circuits` 51%, `wires` 39%,
`frequency` 39%; tower `design` 14%, `line_attachment` 10%, `structure` 8%, `height` 0.1%.

**Recommendation:** PBF as the default source and Overpass only for quick lookups.
Reading PBF needs `pyosmium` (a new dependency) or the `osmium` command-line tool to
filter to `power=*` first.

## 2. Processing, stage by stage

### Stage 1: extract
Keep only power data and preserve the connections between objects.

- Ways: `power=line|cable` (and `minor_line` only if distribution lines are in scope), with
  their **ordered node IDs** and coordinates.
- Nodes: `power=tower|pole|portal|terminal` with tags.
- Substations: `power=substation` ways **and relations** (the relation bug found in the
  downloader matters here).
- Relations: `route=power`, mappers' own grouping of ways into named circuits.

### Stage 2: assemble circuits
1. **Ways → circuits.** Use the `route=power` relation where one exists. Otherwise, join
   ways that share an end node and have the same `voltage`; stop at substations and at
   junctions where three or more ways meet.
2. **Circuit → towers and spans.** Walk the joined node list in order. Nodes tagged
   tower/pole/portal become towers in sequence, and the geometry between two consecutive
   towers is a span. No distance-based matching is needed.
3. **Ends → buses.** Find the substation area each end node falls inside. Bus *location*
   = the end tower's location, because `RoutedTransmissionLine` requires the two to match.
   Bus *identity* comes from the substation's OSM ID and its name/ref.
4. **Split shared structures.** `voltage="230000;115000"` + `circuits=2` means two circuits
   at different voltages on one set of towers: emit one circuit per voltage, sharing the towers.

### Stage 3: enrich
Fill each model field from tags, falling back to defaults, and record where every value came from.

| Field | From tag | Default when missing |
|---|---|---|
| `nominal_voltage_kv` | `voltage` / 1000 | none: skip the circuit |
| `nominal_frequency_hz` | `frequency` | 60 (US) |
| circuit count | `circuits`, else `cables` ÷ 3 | 1 |
| `subconductor_count` | `wires` (single 1, double 2, triple 3, quad 4) | by voltage class |
| conductor | — | catalog pick by voltage class |
| `TowerConfiguration` | `design`, `structure`, `material`, circuit count | library pick by voltage + circuit count |
| `structure_form` | `structure`/`design`/`material` (lattice, H-frame, pole…) | by voltage class |
| `support_role` | `line_management=termination` → terminal; `line_attachment=anchor` → retention | ends terminal, others suspension (optionally retention at sharp bearing changes) |
| tower elevation | — | DEM lookup |
| `earth_resistivity_ohm_m` | — | 100 |

Each circuit carries a `provenance` map, e.g. `{"subconductor_count": "assumed:voltage_class"}`,
so results can say how much of a rating rests on assumptions.

Defaults by voltage class (69 / 115 / 138 / 230 / 345 / 500 / 765 kV) belong in one
table: conductor codeword, bundle, tower design, spacing. Someone with utility experience
should review it, because every unknown ends up resolved there.

## 3. Interface

### Two tiers. Thermal ratings don't need towers.
`thermal_rating(line, weather, route=...)` already accepts a `CrossSectionTransmissionLine`
plus a `RouteGeometry`. So:

- **Tier A (route only):** works for every circuit with a voltage. Cross-section from the
  defaults table + route from the joined ways. Enough for **thermal** and **St. Clair**
  (length = route length).
- **Tier B (towers):** only where towers are mapped along the line. A full
  `RoutedTransmissionLine` with spans. Needed for **sag** (span lengths, elevation differences).

### The handoff record
Stages 1–3 live outside the line models and emit one plain record per circuit.
`r2x-txlines` consumes that record and knows nothing about OSM.

```python
@dataclass(frozen=True)
class CircuitRecord:
    circuit_id: str                          # e.g. "osm:relation/123:230kV" or "osm:ways/1-2-3:230kV"
    voltage_kv: float
    frequency_hz: float
    circuits: int
    subconductor_count: int | None
    route: list[tuple[float, float]]         # (lon, lat), full circuit
    towers: list[TowerRecord]                # empty → Tier A only
    from_bus: BusRecord | None
    to_bus: BusRecord | None
    tags: dict[str, str]                     # raw OSM tags, merged
    provenance: dict[str, str]

@dataclass(frozen=True)
class TowerRecord:
    osm_node_id: int
    lon: float
    lat: float
    elevation_ft: float | None
    structure_form: str | None
    support_role: str | None
    route_index: int                         # position in CircuitRecord.route
```

On-disk format between stages: **GeoParquet**, one row per circuit (route as geometry)
plus a tower table keyed by `circuit_id`. It's cacheable, readable in QGIS, and lets
stages 1–3 rerun without repeating downloads.

### New functions in `r2x-txlines`
```python
# user_api/lines.py: next to cross_section_line
def routed_line(record: CircuitRecord, configuration: TowerConfiguration, *,
                earth_resistivity_ohm_m: float, ...) -> RoutedTransmissionLine: ...

# Tier A helper
def route_of(record: CircuitRecord) -> RouteGeometry: ...
```

End to end, Tier A:
```python
records  = read_circuits("virginia_circuits.parquet")
for rec in records:
    tower    = pick_configuration(rec)                          # defaults table
    line     = cross_section_line(tower, name=rec.circuit_id, voltage_kv=rec.voltage_kv, ...)
    route    = route_of(rec)
    weather  = fetch_weather(weather_points(route), 2012)
    rating   = thermal_rating(line, weather, route=route)
```

### Where the code lives
Open question. Options:
- **A separate package** (extend `openinfra_dl` or start a new one) that owns stages 1–3
  and writes the parquet. `r2x-txlines` only reads `CircuitRecord`. Keeps OSM and PBF
  dependencies out of the calculations package. **Leaning this way.**
- An `r2x-txlines.io.osm` module (`io/` exists and is empty). Simpler to start, but pulls
  in `pyosmium`/geopandas.

## 4. Open questions
- Scope: transmission only (`power=line`), or `minor_line` too? Lowest voltage of interest?
- Who reviews the voltage-class defaults table?
- When OSM and an authoritative dataset (e.g. HIFLD) disagree on voltage or route, which wins?
- Circuits crossing a state line: assemble from a multi-state extract, then deduplicate
  by OSM ID.
- First test: one well-tagged 230 kV Virginia line, Tier A and Tier B, compared against
  its published rating.
