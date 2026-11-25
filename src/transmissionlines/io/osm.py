"""Read OpenStreetMap power data into per-circuit records.

Stages 1 and 2 of ``docs/osm_pipeline.md``: :func:`read_power` pulls power lines,
supports and substations out of an OSM file (``.osm.pbf`` from Geofabrik, or ``.osm``
XML), keeping each way's ordered node IDs. :func:`assemble_circuits` joins ways into
circuits that end at substations or junctions, places towers along each route, and reads
the electrical tags OSM has. :func:`line_from_circuit` turns a record into an
``r2x-txlines`` line (stage 4). Defaults for what OSM lacks come later (stage 3).

Reading and assembling need the ``osm`` extra (``osmium``, ``shapely``); the records and
:func:`line_from_circuit` do not.
"""

from collections import defaultdict
from math import cos, radians
from dataclasses import dataclass, field
from os import PathLike

from transmissionlines.models.assets import CrossSectionTransmissionLine, RoutedTransmissionLine
from transmissionlines.models.common import Bus, GeographicPoint
from transmissionlines.models.configurations import TowerConfiguration
from transmissionlines.models.routing import (
    ElectricalTower, LineSpan, RouteGeometry, StartEnd, span_component_name, tower_component_name,
)

Coordinate = tuple[float, float]  # (longitude, latitude)
LINE_POWER = frozenset({"line", "cable"})
SUPPORT_POWER = frozenset({"tower", "pole", "portal", "terminal"})
# Wiring inside substations, not lines between them.
SUBSTATION_LINES = frozenset({"busbar", "bay", "internal"})
WIRES_PER_PHASE = {"single": 1, "double": 2, "triple": 3, "quad": 4}


@dataclass(frozen=True)
class PowerWay:
    osm_id: int
    tags: dict[str, str]
    node_ids: tuple[int, ...]
    coordinates: tuple[Coordinate, ...]


@dataclass(frozen=True)
class Support:
    osm_id: int
    coordinate: Coordinate
    tags: dict[str, str]


@dataclass(frozen=True)
class Substation:
    osm_ref: str  # "way/123" or "relation/456"
    tags: dict[str, str]
    area: object  # shapely (Multi)Polygon


@dataclass(frozen=True)
class OsmPower:
    ways: list[PowerWay]
    supports: dict[int, Support]
    substations: list[Substation]


@dataclass(frozen=True)
class TowerRecord:
    osm_node_id: int
    coordinate: Coordinate
    route_index: int  # position in CircuitRecord.route
    structure_form: str | None
    support_role: str | None
    elevation_ft: float | None = None
    tags: dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class BusRecord:
    substation_ref: str
    name: str | None
    location: Coordinate  # one point inside the substation, shared by every line ending there


@dataclass(frozen=True)
class CircuitRecord:
    """One circuit at one voltage, ready for enrichment and line building."""

    circuit_id: str
    voltage_kv: float
    way_ids: tuple[int, ...]
    route: tuple[Coordinate, ...]
    towers: tuple[TowerRecord, ...]
    from_bus: BusRecord | None
    to_bus: BusRecord | None
    frequency_hz: float | None
    circuits: int | None
    subconductor_count: int | None
    tags: dict[str, str]  # tags every joined way agrees on
    provenance: dict[str, str]  # field -> where its value came from


def read_power(path: str | PathLike[str]) -> OsmPower:
    """Read power lines, supports and substations (ways and multipolygons) from an OSM file."""
    import osmium
    from shapely import wkb

    ways, supports, substations = [], {}, []
    factory = osmium.geom.WKBFactory()
    processor = (
        osmium.FileProcessor(path)
        .with_locations()
        .with_areas(osmium.filter.TagFilter(("power", "substation")))
        .with_filter(osmium.filter.KeyFilter("power"))
    )
    for obj in processor:
        tags = {tag.k: tag.v for tag in obj.tags}
        power = tags.get("power")
        if obj.is_node() and power in SUPPORT_POWER:
            supports[obj.id] = Support(obj.id, (obj.location.lon, obj.location.lat), tags)
        elif (obj.is_way() and power in LINE_POWER and tags.get("line") not in SUBSTATION_LINES
              and all(n.location.valid() for n in obj.nodes)):
            ways.append(PowerWay(
                obj.id, tags, tuple(n.ref for n in obj.nodes), tuple((n.lon, n.lat) for n in obj.nodes),
            ))
        elif obj.is_area() and power == "substation":
            ref = f"{'way' if obj.from_way() else 'relation'}/{obj.orig_id()}"
            substations.append(Substation(ref, tags, wkb.loads(factory.create_multipolygon(obj), hex=True)))
    return OsmPower(ways, supports, substations)


def assemble_circuits(power: OsmPower, *, substation_tolerance_m: float = 50.0) -> list[CircuitRecord]:
    """Join ways into one record per circuit and voltage.

    Ways carrying the same voltage are joined end to end, except at a node where other
    than two such ways meet or at a substation. A node counts as at a substation when it
    lies within ``substation_tolerance_m`` of its outline, since lines are often mapped
    ending at a portal just outside the fence. A way tagged ``voltage=230000;115000``
    contributes to one circuit at each voltage, which share its route and towers. Ways
    without a ``voltage`` tag are skipped.
    """
    # ponytail: joins only at way ends; a branch tapping mid-way stays its own circuit.
    from shapely import STRtree, Point

    tree = STRtree([s.area for s in power.substations])
    in_substation: dict[int, Substation | None] = {}

    def substation_at(node_id: int, coordinate: Coordinate) -> Substation | None:
        if node_id not in in_substation:
            point = Point(coordinate)
            # ponytail: meters to degrees at the node's latitude; fine away from the poles.
            degrees = substation_tolerance_m / (111_320 * cos(radians(coordinate[1])))
            hits = tree.query(point, predicate="dwithin", distance=degrees)
            nearest = min(hits, key=lambda i: power.substations[i].area.distance(point), default=None)
            in_substation[node_id] = None if nearest is None else power.substations[nearest]
        return in_substation[node_id]

    by_voltage: dict[float, list[PowerWay]] = defaultdict(list)
    for way in power.ways:
        for kv in _voltages_kv(way.tags):
            by_voltage[kv].append(way)

    records = []
    for kv, ways in by_voltage.items():
        for chain in _chains(ways, substation_at):
            record = _record(kv, chain, power.supports, substation_at)
            ends = record.from_bus, record.to_bus
            if not (all(ends) and ends[0].substation_ref == ends[1].substation_ref):  # internal wiring
                records.append(record)
    return records


def route_geometry(record: CircuitRecord) -> RouteGeometry:
    """Return a circuit's route for ``thermal_rating(..., route=...)``."""
    return RouteGeometry(coordinates=list(record.route), source_feature_ids=[f"way/{w}" for w in record.way_ids])


def line_from_circuit(
    record: CircuitRecord,
    configuration: TowerConfiguration,
    *,
    earth_resistivity_ohm_m: float = 100.0,
    default_frequency_hz: float = 60.0,
    default_structure_form: str = "lattice",
) -> RoutedTransmissionLine | CrossSectionTransmissionLine:
    """Build the line a circuit record supports, every tower using ``configuration``.

    A circuit with substations at both ends and mapped towers becomes a
    :class:`RoutedTransmissionLine`. Each end of its route is extended to its
    substation's bus location, where a terminal tower is placed, so every line ending at
    a substation shares one bus. Any other circuit becomes a
    :class:`CrossSectionTransmissionLine`, to be rated along :func:`route_geometry`.
    """
    electrical = dict(
        name=record.circuit_id, nominal_voltage_kv=record.voltage_kv,
        nominal_frequency_hz=record.frequency_hz or default_frequency_hz,
        earth_resistivity_ohm_m=earth_resistivity_ohm_m,
    )
    if not (record.from_bus and record.to_bus and record.towers):
        return CrossSectionTransmissionLine(**electrical, configuration=configuration)

    route = list(record.route)
    offset = 0
    if route[0] != record.from_bus.location:
        route.insert(0, record.from_bus.location)
        offset = 1
    if route[-1] != record.to_bus.location:
        route.append(record.to_bus.location)
    stops: list[tuple[int, TowerRecord | None]] = [(t.route_index + offset, t) for t in record.towers]
    if stops[0][0] != 0:
        stops.insert(0, (0, None))
    if stops[-1][0] != len(route) - 1:
        stops.append((len(route) - 1, None))
    towers = []
    for sequence, (index, mapped) in enumerate(stops):
        longitude, latitude = route[index]
        terminal = sequence in (0, len(stops) - 1)
        towers.append(ElectricalTower(
            name=tower_component_name(record.circuit_id, sequence),
            tower_id=f"node/{mapped.osm_node_id}" if mapped else f"end/{sequence}",
            sequence=sequence,
            location=GeographicPoint(
                longitude_deg=longitude, latitude_deg=latitude,
                elevation_ft=mapped.elevation_ft if mapped else None,
            ),
            configuration=configuration,
            structure_form=(mapped and mapped.structure_form) or default_structure_form,
            support_role="terminal" if terminal else (mapped.support_role or "suspension"),
        ))
    spans = []
    for sequence, (start, end) in enumerate(zip(towers, towers[1:])):
        name = span_component_name(record.circuit_id, sequence)
        spans.append(LineSpan(
            name=name, span_id=f"span/{sequence}", sequence=sequence,
            start_end=StartEnd(name=f"{name}:endpoints", start=start, end=end),
            route_geometry=RouteGeometry(
                coordinates=route[stops[sequence][0]:stops[sequence + 1][0] + 1],
            ),
        ))
    return RoutedTransmissionLine(
        **electrical, spans=spans,
        from_bus=Bus(name=record.from_bus.substation_ref, location=towers[0].location),
        to_bus=Bus(name=record.to_bus.substation_ref, location=towers[-1].location),
    )


def _inside(station: Substation) -> Coordinate:
    point = station.area.representative_point()  # inside the outline, unlike a centroid
    return point.x, point.y


def _voltage_list(tags: dict[str, str]) -> list[float | None]:
    """Return each ``;``-entry of ``voltage`` in kV, None where unreadable, keeping positions."""
    values: list[float | None] = []
    for value in _split(tags.get("voltage")):
        try:
            kv = float(value) / 1000
        except ValueError:
            kv = None
        values.append(kv if kv and kv > 0 else None)
    return values


def _voltages_kv(tags: dict[str, str]) -> list[float]:
    """Return the distinct voltages a way carries; ``69000;69000`` is two circuits at one."""
    return list(dict.fromkeys(kv for kv in _voltage_list(tags) if kv is not None))


def _split(value: str | None) -> list[str]:
    return [part.strip() for part in value.split(";")] if value else []


def _chains(ways, substation_at) -> list[list[tuple[PowerWay, bool]]]:
    """Group ways into chains of (way, reversed) running end to end."""
    at_node: dict[int, list[int]] = defaultdict(list)
    for index, way in enumerate(ways):
        at_node[way.node_ids[0]].append(index)
        at_node[way.node_ids[-1]].append(index)

    used, chains = set(), []
    for start in range(len(ways)):
        if start in used:
            continue
        used.add(start)
        chain = [(start, False)]
        for forward in (True, False):  # grow from the chain's last end, then its first
            while True:
                index, flipped = chain[-1] if forward else chain[0]
                end = -1 if forward != flipped else 0
                node = ways[index].node_ids[end]
                touching = at_node[node]
                if len(touching) != 2 or substation_at(node, ways[index].coordinates[end]) is not None:
                    break
                other = touching[0] if touching[1] == index else touching[1]
                if other in used:  # closed loop, or a way ending on itself
                    break
                used.add(other)
                # Orient the new way so it continues from the shared node.
                starts_there = ways[other].node_ids[0] == node
                if forward:
                    chain.append((other, not starts_there))
                else:
                    chain.insert(0, (other, starts_there))
        chains.append([(ways[i], flipped) for i, flipped in chain])
    return chains


def _record(kv, chain, supports, substation_at) -> CircuitRecord:
    node_ids: list[int] = []
    route: list[Coordinate] = []
    for way, flipped in chain:
        ids = way.node_ids[::-1] if flipped else way.node_ids
        coords = way.coordinates[::-1] if flipped else way.coordinates
        skip = 1 if node_ids else 0  # the shared node is already there
        node_ids.extend(ids[skip:])
        route.extend(coords[skip:])

    towers = tuple(
        TowerRecord(
            osm_node_id=node_id, coordinate=coordinate, route_index=index,
            structure_form=_structure_form(supports[node_id].tags),
            support_role=_support_role(supports[node_id].tags), tags=supports[node_id].tags,
        )
        for index, (node_id, coordinate) in enumerate(zip(node_ids, route))
        if node_id in supports
    )
    buses = [
        None if (station := substation_at(node_ids[i], route[i])) is None
        else BusRecord(station.osm_ref, station.tags.get("name"), _inside(station))
        for i in (0, -1)
    ]
    first_tags = chain[0][0].tags
    tags = {k: v for k, v in first_tags.items() if all(w.tags.get(k) == v for w, _ in chain)}
    voltages = _voltage_list(tags) or [kv]  # joined ways that disagree on voltage keep none
    positions = [i for i, value in enumerate(voltages) if value == kv] or [0]

    def at(key, parse):
        return _values_at(tags, key, positions, len(voltages), parse)

    provenance = {"voltage_kv": "osm:voltage"}
    frequency = next(iter(at("frequency", float)), None)
    circuits = sum(at("circuits", int)) or None
    if circuits is None and (cables := sum(at("cables", int))) and cables % 3 == 0:
        circuits, provenance["circuits"] = cables // 3, "osm:cables"
    wires = next(iter(at("wires", WIRES_PER_PHASE.get)), None)
    for name, value, tag in (("frequency_hz", frequency, "frequency"), ("circuits", circuits, "circuits"),
                             ("subconductor_count", wires, "wires")):
        if value is not None:
            provenance.setdefault(name, f"osm:{tag}")

    return CircuitRecord(
        circuit_id=f"osm:way/{chain[0][0].osm_id}:{kv:g}kV",
        voltage_kv=kv, way_ids=tuple(w.osm_id for w, _ in chain), route=tuple(route), towers=towers,
        from_bus=buses[0], to_bus=buses[1], frequency_hz=frequency, circuits=circuits,
        subconductor_count=wires, tags=tags, provenance=provenance,
    )


def _values_at(tags, key, positions, count, parse) -> list:
    """Read one voltage's entries of a ``;``-list aligned with ``voltage``; all or nothing."""
    parts = _split(tags.get(key))
    if len(parts) == count:
        raw = [parts[i] for i in positions]
    elif len(parts) == 1 and count == 1:
        raw = parts
    else:
        return []
    try:
        values = [parse(part) for part in raw]
    except ValueError:
        return []
    return values if None not in values else []


def _structure_form(tags: dict[str, str]) -> str | None:
    design = tags.get("design", "").lower().replace("_", "-")
    if "h-frame" in design:
        return "h_frame"
    if "y-frame" in design:
        return "y_frame"
    structure = tags.get("structure")
    if structure == "lattice":
        return "lattice"
    if structure in {"tubular", "solid"} or tags.get("power") == "pole":
        return "pole"
    return None


def _support_role(tags: dict[str, str]) -> str | None:
    if tags.get("line_management") == "termination":
        return "terminal"
    return {"anchor": "retention", "suspension": "suspension"}.get(tags.get("line_attachment", ""))


__all__ = [
    "BusRecord", "CircuitRecord", "OsmPower", "PowerWay", "Substation", "Support", "TowerRecord",
    "assemble_circuits", "line_from_circuit", "read_power", "route_geometry",
]
