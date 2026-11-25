from xml.sax.saxutils import quoteattr

import pytest

pytest.importorskip("osmium")

from transmissionlines.io.osm import (  # noqa: E402
    assemble_circuits, line_from_circuit, read_power, route_geometry,
)
from transmissionlines.models.assets import (  # noqa: E402
    CrossSectionTransmissionLine, RoutedTransmissionLine,
)

from transmissionlines.builders.system import assemble_line_into_system  # noqa: E402
from transmissionlines.models.common import Bus  # noqa: E402
from transmissionlines.system import TransmissionLineSystem  # noqa: E402

from test.models.test_input_graph import _configuration  # noqa: E402

# Substation A is a closed way around node 1; substation B is a multipolygon relation
# around node 5. The line 1-2-3-4-5 is split at 3 into two ways, the second drawn backwards.
_NODES = {
    1: (-77.000, 38.0), 2: (-76.99, 38.0), 3: (-76.98, 38.0), 4: (-76.97, 38.0), 5: (-76.96, 38.0),
    40: (-77.001, 37.999), 41: (-76.9995, 37.999), 42: (-76.9995, 38.001), 43: (-77.001, 38.001),
    30: (-76.961, 37.999), 31: (-76.959, 37.999), 32: (-76.959, 38.001), 33: (-76.961, 38.001),
    6: (-76.97, 38.01), 7: (-76.98, 38.02), 8: (-76.975, 38.0005), 9: (-76.975, 37.9995),
}
_TOWER = {"power": "tower", "structure": "lattice", "line_attachment": "suspension"}
_NODE_TAGS = {2: _TOWER, 3: {"power": "tower", "line_attachment": "anchor"}, 4: _TOWER}
_LINE = {"power": "line", "voltage": "230000", "circuits": "1", "wires": "double", "frequency": "60"}


def _tags(tags: dict[str, str]) -> str:
    return "".join(f'<tag k="{k}" v={quoteattr(v)}/>' for k, v in tags.items())


def _way(way_id: int, nodes: list[int], tags: dict[str, str]) -> tuple[int, str]:
    refs = "".join(f'<nd ref="{n}"/>' for n in nodes)
    return way_id, f'<way id="{way_id}" version="1">{refs}{_tags(tags)}</way>'


def _osm(tmp_path, *ways: tuple[int, str], alpha: dict[str, str] | None = None) -> str:
    nodes = "".join(
        f'<node id="{i}" version="1" lon="{lon}" lat="{lat}">{_tags(_NODE_TAGS.get(i, {}))}</node>'
        for i, (lon, lat) in _NODES.items()
    )
    substations = (
        _way(50, [40, 41, 42, 43, 40], {"power": "substation", "name": "Alpha", **(alpha or {})}),
        _way(20, [30, 31, 32], {}), _way(21, [32, 33, 30], {}),
    )
    all_ways = "".join(xml for _, xml in sorted((*substations, *ways)))  # OSM files are sorted by ID
    relation = (
        '<relation id="60" version="1"><member type="way" ref="20" role="outer"/>'
        '<member type="way" ref="21" role="outer"/>'
        + _tags({"type": "multipolygon", "power": "substation", "name": "Bravo"}) + "</relation>"
    )
    path = tmp_path / "grid.osm"
    path.write_text(f'<osm version="0.6">{nodes}{all_ways}{relation}</osm>')
    return str(path)


def test_split_line_joins_into_one_circuit_between_substations(tmp_path) -> None:
    path = _osm(tmp_path, _way(10, [1, 2, 3], _LINE), _way(11, [5, 4, 3], _LINE))

    (circuit,) = assemble_circuits(read_power(path))

    assert circuit.route == tuple(_NODES[i] for i in (1, 2, 3, 4, 5))
    assert [(t.osm_node_id, t.route_index) for t in circuit.towers] == [(2, 1), (3, 2), (4, 3)]
    assert [t.support_role for t in circuit.towers] == ["suspension", "retention", "suspension"]
    assert circuit.towers[0].structure_form == "lattice"
    assert (circuit.from_bus.name, circuit.from_bus.substation_ref) == ("Alpha", "way/50")
    assert (circuit.to_bus.name, circuit.to_bus.substation_ref) == ("Bravo", "relation/60")
    lon, lat = circuit.to_bus.location
    assert -76.961 < lon < -76.959 and 37.999 < lat < 38.001  # inside substation Bravo
    assert (circuit.voltage_kv, circuit.frequency_hz, circuit.circuits, circuit.subconductor_count) == (
        230, 60, 1, 2,
    )
    assert circuit.provenance["subconductor_count"] == "osm:wires"
    assert route_geometry(circuit).coordinates[-1] == _NODES[5]


def test_shared_structure_gives_one_circuit_per_voltage(tmp_path) -> None:
    shared = {"power": "line", "voltage": "230000;115000", "cables": "3;3", "wires": "double;single"}
    path = _osm(tmp_path, _way(10, [1, 2, 3, 4, 5], shared))

    circuits = {c.voltage_kv: c for c in assemble_circuits(read_power(path))}

    assert circuits.keys() == {230, 115}
    assert (circuits[230].subconductor_count, circuits[115].subconductor_count) == (2, 1)
    assert circuits[115].circuits == 1 and circuits[115].provenance["circuits"] == "osm:circuits"
    assert circuits[230].towers == circuits[115].towers


def test_three_way_junction_ends_each_circuit(tmp_path) -> None:
    path = _osm(
        tmp_path, _way(10, [1, 2, 3], _LINE), _way(11, [3, 4, 5], _LINE), _way(12, [3, 6, 7], _LINE),
    )

    circuits = assemble_circuits(read_power(path))

    assert sorted(c.way_ids for c in circuits) == [(10,), (11,), (12,)]
    branch = next(c for c in circuits if c.way_ids == (12,))
    assert branch.to_bus is None and branch.subconductor_count == 2


def test_lines_without_voltage_are_skipped(tmp_path) -> None:
    path = _osm(tmp_path, _way(10, [1, 2, 3, 4, 5], {"power": "line"}))

    assert assemble_circuits(read_power(path)) == []


def test_line_ending_just_outside_a_substation_reaches_it(tmp_path) -> None:
    # Node 2 is ~800 m east of substation Alpha's fence.
    path = _osm(tmp_path, _way(10, [2, 3, 4, 5], _LINE))

    (near,) = assemble_circuits(read_power(path), substation_tolerance_m=1000)
    (strict,) = assemble_circuits(read_power(path), substation_tolerance_m=10)

    assert near.from_bus.name == "Alpha" and strict.from_bus is None


def test_wiring_inside_a_substation_is_not_a_circuit(tmp_path) -> None:
    bay = {**_LINE, "line": "bay"}
    path = _osm(tmp_path, _way(10, [40, 1], bay), _way(11, [40, 42], _LINE))

    assert assemble_circuits(read_power(path)) == []


def test_circuit_between_substations_becomes_a_routed_line(tmp_path) -> None:
    path = _osm(tmp_path, _way(10, [1, 2, 3], _LINE), _way(11, [5, 4, 3], _LINE))
    (circuit,) = assemble_circuits(read_power(path))

    line = line_from_circuit(circuit, _configuration())

    assert isinstance(line, RoutedTransmissionLine)
    towers = [line.spans[0].start_end.start, *(span.start_end.end for span in line.spans)]
    # Terminals sit at each substation's bus location, reached from route ends 1 and 5.
    assert [t.tower_id for t in towers] == ["end/0", "node/2", "node/3", "node/4", "end/4"]
    assert [t.support_role for t in towers] == ["terminal", "suspension", "retention", "suspension", "terminal"]
    assert line.spans[0].route_geometry.coordinates == [circuit.from_bus.location, _NODES[1], _NODES[2]]
    assert line.spans[1].route_geometry.coordinates == [_NODES[2], _NODES[3]]
    assert line.from_bus.location == towers[0].location
    assert (line.from_bus.name, line.to_bus.name) == ("way/50", "relation/60")
    assert line.nominal_voltage_kv == 230


def test_lines_ending_at_one_substation_share_its_bus(tmp_path) -> None:
    shared = {**_LINE, "voltage": "230000;115000", "circuits": "1;1", "wires": "double;single"}
    path = _osm(tmp_path, _way(10, [1, 2, 3, 4, 5], shared))
    lines = [line_from_circuit(c, _configuration()) for c in assemble_circuits(read_power(path))]
    system = TransmissionLineSystem(name="shared")

    for line in lines:
        assemble_line_into_system(system, line)

    assert len(lines) == 2 and all(isinstance(line, RoutedTransmissionLine) for line in lines)
    assert sorted(bus.name for bus in system.get_components(Bus)) == ["relation/60", "way/50"]


def test_open_ended_circuit_becomes_a_cross_section(tmp_path) -> None:
    path = _osm(tmp_path, _way(10, [1, 2, 3], _LINE), _way(11, [3, 4, 5], _LINE), _way(12, [3, 6, 7], _LINE))
    branch = next(c for c in assemble_circuits(read_power(path)) if c.way_ids == (12,))

    line = line_from_circuit(branch, _configuration())

    assert isinstance(line, CrossSectionTransmissionLine)
    assert route_geometry(branch).coordinates[0] == _NODES[3]


def test_two_circuits_at_one_voltage_are_one_record(tmp_path) -> None:
    double = {"power": "line", "voltage": "69000;69000", "cables": "3;3", "wires": "single;single"}
    path = _osm(tmp_path, _way(10, [1, 2, 3, 4, 5], double))

    (circuit,) = assemble_circuits(read_power(path))

    assert (circuit.voltage_kv, circuit.circuits, circuit.subconductor_count) == (69, 2, 1)


def test_single_phase_ways_fold_into_their_circuit(tmp_path) -> None:
    phase = {**_LINE, "cables": "1", "circuits": "1/3"}
    path = _osm(
        tmp_path, _way(10, [1, 2, 3], _LINE), _way(11, [3, 8, 4], phase), _way(12, [3, 4], phase),
        _way(13, [3, 9, 4], phase), _way(14, [4, 5], _LINE),
    )

    (circuit,) = assemble_circuits(read_power(path))

    assert (circuit.from_bus.name, circuit.to_bus.name) == ("Alpha", "Bravo")
    assert circuit.way_ids == (10, 11, 14) and circuit.circuits == 1  # 11 is the middle-length phase


def test_double_circuit_splitting_and_rejoining_stays_one_record(tmp_path) -> None:
    double = {**_LINE, "cables": "6", "circuits": "2", "name": "X"}
    single = {**_LINE, "cables": "3", "circuits": "1", "name": "X"}
    path = _osm(
        tmp_path, _way(10, [1, 2, 3], double), _way(11, [3, 8, 4], single), _way(12, [3, 9, 4], single),
        _way(13, [4, 5], double),
    )

    (circuit,) = assemble_circuits(read_power(path))

    assert (circuit.from_bus.name, circuit.to_bus.name, circuit.circuits) == ("Alpha", "Bravo", 2)


def test_double_circuit_splitting_into_named_circuits_follows_names(tmp_path) -> None:
    path = _osm(
        tmp_path, _way(10, [1, 2, 3], {**_LINE, "cables": "6", "circuits": "2", "name": "A & B"}),
        _way(11, [3, 6, 7], {**_LINE, "name": "B"}), _way(12, [3, 4, 5], {**_LINE, "name": "A"}),
    )

    circuits = {c.way_ids: c for c in assemble_circuits(read_power(path))}

    assert circuits.keys() == {(10, 12), (10, 11)}
    assert circuits[(10, 12)].to_bus.name == "Bravo" and circuits[(10, 11)].to_bus is None
    assert len({c.circuit_id for c in circuits.values()}) == 2


def test_repeated_voltage_with_one_total_counts_both_circuits(tmp_path) -> None:
    path = _osm(tmp_path, _way(10, [1, 2, 3, 4, 5], {**_LINE, "voltage": "230000;230000", "circuits": "2"}))

    (circuit,) = assemble_circuits(read_power(path))

    assert circuit.circuits == 2 and circuit.provenance["circuits"] == "osm:circuits"


@pytest.mark.parametrize(("alpha", "reach_m", "reached"), [
    ({}, 1000, True),  # no voltage tag: accepted
    ({"voltage": "230000;69000"}, 1000, True),
    ({"voltage": "115000"}, 1000, False),  # does not carry the line's voltage
    ({}, 500, False),  # beyond reach
])
def test_dead_end_snaps_to_a_nearby_substation_with_its_voltage(tmp_path, alpha, reach_m, reached) -> None:
    # Node 2, the line's dead end, is ~830 m east of substation Alpha's fence.
    path = _osm(tmp_path, _way(10, [2, 3, 4, 5], _LINE), alpha=alpha)

    (circuit,) = assemble_circuits(read_power(path), dead_end_reach_m=reach_m)

    assert (circuit.from_bus is not None) == reached
    if reached:
        assert circuit.from_bus.name == "Alpha" and 700 < circuit.from_bus.distance_m < 900
    assert circuit.to_bus.distance_m == 0  # node 5 is inside Bravo


def test_junction_near_a_substation_is_not_its_end(tmp_path) -> None:
    # Junction 2 is a tap within reach of Alpha, but not a dead end, so circuits end there.
    path = _osm(tmp_path, _way(10, [2, 3, 4, 5], _LINE), _way(11, [2, 6], _LINE), _way(12, [7, 2], _LINE))

    circuits = assemble_circuits(read_power(path), dead_end_reach_m=1000)

    ends_at_junction = [
        bus for c in circuits for bus, end in ((c.from_bus, c.route[0]), (c.to_bus, c.route[-1]))
        if end == _NODES[2]
    ]
    assert len(ends_at_junction) == 3 and ends_at_junction == [None, None, None]
