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
from math import asin, cos, hypot, radians, sin, sqrt
from dataclasses import dataclass, field, replace
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
    distance_m: float = 0.0  # from the circuit's end to the substation outline; review long ones


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


def read_power(*paths: str | PathLike[str]) -> OsmPower:
    """Read power lines, supports and substations (ways and multipolygons) from OSM files.

    Several files, such as neighbouring state extracts, are combined; objects crossing
    their shared border appear in each and are kept once.
    """
    import osmium
    from shapely import wkb

    ways, supports, substations = {}, {}, {}
    factory = osmium.geom.WKBFactory()
    for path in paths:
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
                ways[obj.id] = PowerWay(
                    obj.id, tags, tuple(n.ref for n in obj.nodes), tuple((n.lon, n.lat) for n in obj.nodes),
                )
            elif obj.is_area() and power == "substation":
                ref = f"{'way' if obj.from_way() else 'relation'}/{obj.orig_id()}"
                substations[ref] = Substation(ref, tags, wkb.loads(factory.create_multipolygon(obj), hex=True))
    return OsmPower(list(ways.values()), supports, list(substations.values()))


def assemble_circuits(
    power: OsmPower, *, substation_tolerance_m: float = 50.0, dead_end_reach_m: float = 400.0,
) -> list[CircuitRecord]:
    """Join ways into one record per circuit route and voltage.

    Per voltage:

    1. Ways mapping single phases (``cables=1``, ``circuits=1/3``) that share both ends
       are folded into one way carrying the whole circuit.
    2. Each way carries as many circuits as its ``circuits`` (or ``cables`` / 3) tag says,
       else one. Circuits are followed through each node: the ways meeting there are split
       into two sides by direction, and when both sides carry as many circuits, each
       circuit continues into one on the far side, preferring ways with matching names.
       A double-circuit way splitting into two single-circuit ways passes both through;
       a tap, where the circuit counts do not balance, ends every circuit there.
    3. Circuits with the same two ends that share most of their ways become one record
       whose ``circuits`` counts them.

    Circuits end at substations: a node within ``substation_tolerance_m`` of an outline,
    since lines are often mapped ending at a portal just outside the fence. A line does not
    just stop, so a dead end (no other way of its voltage continues) further out ends at the
    nearest substation within ``dead_end_reach_m`` that carries its voltage or has no
    ``voltage`` tag; :attr:`BusRecord.distance_m` records how far. Junctions keep the strict
    tolerance, so a split near a substation is not mistaken for its end. A way tagged
    ``voltage=230000;115000`` contributes at each voltage. Ways without a ``voltage`` tag
    are skipped.
    """
    # ponytail: joins only at way ends; a branch tapping mid-way stays its own circuit.
    from shapely import STRtree, Point

    tree = STRtree([s.area for s in power.substations])
    in_substation: dict[int, tuple[Substation | None, float]] = {}

    def nearest(coordinate: Coordinate, reach_m: float, accept=lambda station: True):
        # ponytail: meters to degrees at the node's latitude; fine away from the poles.
        degrees = reach_m / (111_320 * cos(radians(coordinate[1])))
        found = [
            (_meters_to(power.substations[i].area, coordinate), power.substations[i])
            for i in tree.query(Point(coordinate), predicate="dwithin", distance=degrees)
        ]
        found = [(meters, station) for meters, station in found if meters <= reach_m and accept(station)]
        meters, station = min(found, key=lambda pair: pair[0], default=(0.0, None))
        return station, meters

    def strict(node_id: int, coordinate: Coordinate) -> tuple[Substation | None, float]:
        if node_id not in in_substation:
            in_substation[node_id] = nearest(coordinate, substation_tolerance_m)
        return in_substation[node_id]

    def substation_at(node_id: int, coordinate: Coordinate) -> Substation | None:
        return strict(node_id, coordinate)[0]

    by_voltage: dict[float, list[PowerWay]] = defaultdict(list)
    for way in power.ways:
        for kv in _voltages_kv(way.tags):
            by_voltage[kv].append(way)

    records = []
    for kv, ways in by_voltage.items():
        ways = _fold_phases(ways, substation_at)
        ends_at = defaultdict(int)
        for way in ways:
            ends_at[way.node_ids[0]] += 1
            ends_at[way.node_ids[-1]] += 1

        def bus_at(node_id: int, coordinate: Coordinate, kv=kv, ends_at=ends_at) -> BusRecord | None:
            station, meters = strict(node_id, coordinate)
            if station is None and ends_at[node_id] == 1:
                station, meters = nearest(
                    coordinate, dead_end_reach_m,
                    lambda s: not _voltages_kv(s.tags) or kv in _voltages_kv(s.tags),
                )
            if station is None:
                return None
            return BusRecord(station.osm_ref, station.tags.get("name"), _inside(station), round(meters))

        for chain, count in _merge_parallel(_slot_chains(ways, kv, substation_at), ways, substation_at):
            record = _record(kv, chain, power.supports, bus_at, count)
            ends = record.from_bus, record.to_bus
            if not (all(ends) and ends[0].substation_ref == ends[1].substation_ref):  # internal wiring
                records.append(record)
    seen: dict[str, int] = defaultdict(int)
    for index, record in enumerate(records):  # circuits sharing first and last way
        seen[record.circuit_id] += 1
        if seen[record.circuit_id] > 1:
            records[index] = replace(record, circuit_id=f"{record.circuit_id}#{seen[record.circuit_id]}")
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


def _meters_to(area, coordinate: Coordinate) -> float:
    """Return the distance from a point to an outline in meters; 0 inside it."""
    from shapely import Point
    from shapely.ops import nearest_points

    point = Point(coordinate)
    if area.contains(point):
        return 0.0
    # ponytail: nearest in degrees, not meters; close enough at substation scale.
    edge = nearest_points(area, point)[0]
    return _haversine_m(coordinate, (edge.x, edge.y))


def _haversine_m(a: Coordinate, b: Coordinate) -> float:
    (lon1, lat1), (lon2, lat2) = a, b
    h = sin(radians(lat2 - lat1) / 2) ** 2 + cos(radians(lat1)) * cos(radians(lat2)) * sin(
        radians(lon2 - lon1) / 2) ** 2
    return 2 * 6_371_000 * asin(sqrt(h))


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


def _is_phase(tags: dict[str, str]) -> bool:
    return tags.get("cables") == "1" or any(part.endswith("/3") for part in _split(tags.get("circuits")))


def _end_key(way: PowerWay, end: int, substation_at) -> int | str:
    """Return a way end's node, or its substation: phases often land on separate nodes inside one."""
    station = substation_at(way.node_ids[end], way.coordinates[end])
    return station.osm_ref if station else way.node_ids[end]


def _fold_phases(ways: list[PowerWay], substation_at) -> list[PowerWay]:
    """Replace single-phase ways sharing both ends with the middle one, as a whole circuit."""
    kept, groups = [], defaultdict(list)
    for way in ways:
        if _is_phase(way.tags):
            groups[frozenset((_end_key(way, 0, substation_at), _end_key(way, -1, substation_at)))].append(way)
        else:
            kept.append(way)
    for group in groups.values():
        if len(group) == 1:
            kept.extend(group)
            continue
        group.sort(key=lambda way: _length_deg(way.coordinates))
        middle = group[len(group) // 2]
        circuits = max(1, round(len(group) / 3))  # two of three phases mapped is still one circuit
        tags = {**middle.tags, "cables": str(3 * circuits), "circuits": str(circuits)}
        kept.append(PowerWay(middle.osm_id, tags, middle.node_ids, middle.coordinates))
    return kept


def _length_deg(coordinates) -> float:
    return sum(hypot(x1 - x0, y1 - y0) for (x0, y0), (x1, y1) in zip(coordinates, coordinates[1:]))


def _circuit_count(tags: dict[str, str], kv: float) -> int | None:
    """Return the circuits a way carries at ``kv``, from ``circuits`` or else ``cables`` / 3."""
    voltages = _voltage_list(tags) or [kv]
    positions = [i for i, value in enumerate(voltages) if value == kv] or [0]
    if circuits := sum(_values_at(tags, "circuits", positions, len(voltages), int)):
        return circuits
    cables = sum(_values_at(tags, "cables", positions, len(voltages), int))
    return cables // 3 if cables and cables % 3 == 0 else None


def _direction(way: PowerWay, end: int, reach_m: float = 500.0) -> tuple[float, float]:
    """Return the unit vector from a way's end toward its point ``reach_m`` along, or its far end.

    Looking past the first span keeps a circuit's general heading where ways fan out at a
    split, as the two circuits of a double-circuit line do around a dead-end tower.
    """
    coordinates = way.coordinates if end == 0 else way.coordinates[::-1]
    x0, y0 = coordinates[0]
    scale = cos(radians(y0))
    travelled, (x1, y1) = 0.0, coordinates[-1]
    for (xa, ya), (xb, yb) in zip(coordinates, coordinates[1:]):
        travelled += hypot((xb - xa) * scale, yb - ya) * 111_320
        if travelled >= reach_m:
            x1, y1 = xb, yb
            break
    dx, dy = (x1 - x0) * scale, y1 - y0
    norm = hypot(dx, dy) or 1.0
    return dx / norm, dy / norm


def _sides(ways, ends, slots) -> tuple[list, list] | None:
    """Split way ends meeting at a node into the two directions circuits run through it.

    Of the splits whose two sides carry as many circuits, take the one whose sides point
    most nearly opposite. None when no split balances: a tap.
    """
    if len(ends) == 2:
        return [ends[0]], [ends[1]]
    directions = [_direction(ways[i], end) for i, end in ends]
    best, best_score = None, None
    for mask in range(1, 2 ** (len(ends) - 1)):  # the last end always sits on the second side
        one = [k for k in range(len(ends)) if mask >> k & 1]
        other = [k for k in range(len(ends)) if not mask >> k & 1]
        if sum(slots[ends[k][0]] for k in one) != sum(slots[ends[k][0]] for k in other):
            continue
        score = sum(
            directions[a][0] * directions[b][0] + directions[a][1] * directions[b][1] for a in one for b in other
        ) / (len(one) * len(other))
        if best_score is None or score < best_score:
            best, best_score = ([ends[k] for k in one], [ends[k] for k in other]), score
    return best


def _pair_by_name(left: list, right: list, ways) -> list[tuple]:
    """Pair circuit slots across a node, same names first, then names containing the other."""
    def score(x, y) -> int:
        p, q = ways[x[0]].tags.get("name"), ways[y[0]].tags.get("name")
        if not p or not q:
            return 0
        return 2 if p == q else 1 if p in q or q in p else 0

    pairs, free = [], list(right)
    for x in left:
        if not free:
            break
        y = max(free, key=lambda y: score(x, y))
        free.remove(y)
        pairs.append((x, y))
    return pairs


def _slot_chains(ways: list[PowerWay], kv: float, substation_at) -> list[list[tuple[int, int, bool]]]:
    """Follow each circuit slot through nodes; chains of (way index, slot, reversed)."""
    slots = [_circuit_count(way.tags, kv) or 1 for way in ways]
    at_node: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for index, way in enumerate(ways):
        at_node[way.node_ids[0]].append((index, 0))
        at_node[way.node_ids[-1]].append((index, -1))

    matches: dict[int, dict] = {}

    def match(node: int) -> dict:
        if node not in matches:
            ends, result = at_node[node], {}
            index, end = ends[0]
            sides = None
            if len(ends) >= 2 and substation_at(node, ways[index].coordinates[end]) is None:
                sides = _sides(ways, ends, slots)
            if sides:
                left = [(i, e, k) for i, e in sides[0] for k in range(slots[i])]
                right = [(i, e, k) for i, e in sides[1] for k in range(slots[i])]
                for x, y in _pair_by_name(left, right, ways):
                    result[x], result[y] = y, x
            matches[node] = result
        return matches[node]

    used, chains = set(), []
    for start in range(len(ways)):
        for slot in range(slots[start]):
            if (start, slot) in used:
                continue
            used.add((start, slot))
            chain = [(start, slot, False)]
            for forward in (True, False):  # grow from the chain's last end, then its first
                while True:
                    index, k, flipped = chain[-1] if forward else chain[0]
                    end = -1 if forward != flipped else 0
                    following = match(ways[index].node_ids[end]).get((index, end, k))
                    if following is None or following[::2] in used:
                        break
                    j, j_end, j_slot = following
                    used.add((j, j_slot))
                    if forward:  # enter way j at the shared node
                        chain.append((j, j_slot, j_end != 0))
                    else:  # way j must finish at the shared node
                        chain.insert(0, (j, j_slot, j_end == 0))
            chains.append(chain)
    return chains


def _merge_parallel(chains, ways, substation_at) -> list[tuple[list[tuple[PowerWay, bool]], int]]:
    """Merge circuits with the same ends sharing most of their length; return (chain, circuit count).

    A double-circuit line whose circuits split onto separate ways for a few spans is one
    corridor: its two circuits share the double-circuit ways for most of their length.
    """
    # ponytail: "most" = at least half the shorter circuit's length; tune on real corridors.
    lengths = [_length_deg(way.coordinates) for way in ways]
    merged: list[list] = []  # [ends, way indexes, length, chain, count]
    for chain in chains:
        first, last = chain[0], chain[-1]
        ends = frozenset((
            _end_key(ways[first[0]], -1 if first[2] else 0, substation_at),
            _end_key(ways[last[0]], 0 if last[2] else -1, substation_at),
        ))
        indexes = {i for i, _, _ in chain}
        length = sum(lengths[i] for i in indexes)
        for group in merged:
            shared = sum(lengths[i] for i in group[1] & indexes)
            if group[0] == ends and shared * 2 >= min(group[2], length):
                group[4] += 1
                break
        else:
            merged.append([ends, indexes, length, chain, 1])
    return [([(ways[i], flipped) for i, _, flipped in chain], count) for *_, chain, count in merged]


def _record(kv, chain, supports, bus_at, circuits) -> CircuitRecord:
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
    buses = [bus_at(node_ids[i], route[i]) for i in (0, -1)]
    first_tags = chain[0][0].tags
    tags = {k: v for k, v in first_tags.items() if all(w.tags.get(k) == v for w, _ in chain)}
    voltages = _voltage_list(tags) or [kv]  # joined ways that disagree on voltage keep none
    positions = [i for i, value in enumerate(voltages) if value == kv] or [0]

    def at(key, parse):
        return _values_at(tags, key, positions, len(voltages), parse)

    provenance = {"voltage_kv": "osm:voltage"}
    # Counted from the ways' circuit tags; a way without them is assumed to carry one.
    tagged = all(_circuit_count(way.tags, kv) is not None for way, _ in chain)
    provenance["circuits"] = "osm:circuits" if tagged else "assumed:untagged ways carry one"
    frequency = next(iter(at("frequency", float)), None)
    wires = next(iter(at("wires", WIRES_PER_PHASE.get)), None)
    for name, value, tag in (("frequency_hz", frequency, "frequency"), ("subconductor_count", wires, "wires")):
        if value is not None:
            provenance[name] = f"osm:{tag}"

    return CircuitRecord(
        circuit_id=f"osm:way/{chain[0][0].osm_id}-{chain[-1][0].osm_id}:{kv:g}kV",
        voltage_kv=kv, way_ids=tuple(w.osm_id for w, _ in chain), route=tuple(route), towers=towers,
        from_bus=buses[0], to_bus=buses[1], frequency_hz=frequency, circuits=circuits,
        subconductor_count=wires, tags=tags, provenance=provenance,
    )


def _values_at(tags, key, positions, count, parse) -> list:
    """Read one voltage's entries of a ``;``-list aligned with ``voltage``; all or nothing."""
    parts = _split(tags.get(key))
    if len(parts) == count:
        raw = [parts[i] for i in positions]
    elif len(parts) == 1 and len(positions) == count:  # one value for e.g. ``230000;230000``
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
