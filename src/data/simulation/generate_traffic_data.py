import os
import numpy as np
import xml.etree.ElementTree as ET
from xml.dom import minidom


def prettify(elem):
    """Return a pretty-printed XML string for the Element."""
    rough_string = ET.tostring(elem, "utf-8")
    reparsed = minidom.parseString(rough_string)
    return reparsed.toprettyxml(indent="  ")


def create_low_traffic_scenario():
    """Creates a low traffic volume scenario"""
    routes = ET.Element("routes")
    routes.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    routes.set(
        "xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd"
    )

    # Vehicle types
    vtype1 = ET.SubElement(routes, "vType")
    vtype1.set("id", "passenger")
    vtype1.set("length", "5.0")
    vtype1.set("minGap", "2.5")
    vtype1.set("maxSpeed", "15.0")
    vtype1.set("accel", "2.0")
    vtype1.set("decel", "4.5")
    vtype1.set("sigma", "0.5")

    vtype2 = ET.SubElement(routes, "vType")
    vtype2.set("id", "truck")
    vtype2.set("length", "7.5")
    vtype2.set("minGap", "3.0")
    vtype2.set("maxSpeed", "13.0")
    vtype2.set("accel", "1.5")
    vtype2.set("decel", "3.5")
    vtype2.set("sigma", "0.5")

    # Define all possible routes
    route_definitions = [
        ("route_ns", "north_to_center center_to_south"),
        ("route_ne", "north_to_center center_to_east"),
        ("route_nw", "north_to_center center_to_west"),
        ("route_sn", "south_to_center center_to_north"),
        ("route_se", "south_to_center center_to_east"),
        ("route_sw", "south_to_center center_to_west"),
        ("route_ew", "east_to_center center_to_west"),
        ("route_en", "east_to_center center_to_north"),
        ("route_es", "east_to_center center_to_south"),
        ("route_we", "west_to_center center_to_east"),
        ("route_wn", "west_to_center center_to_north"),
        ("route_ws", "west_to_center center_to_south"),
    ]

    # Add all routes
    for route_id, edges in route_definitions:
        route = ET.SubElement(routes, "route")
        route.set("id", route_id)
        route.set("edges", edges)

    # Define low traffic flows (probability < 0.1)
    flow_definitions = [
        ("flow_ns", "passenger", "route_ns", "0.05"),
        ("flow_sn", "passenger", "route_sn", "0.05"),
        ("flow_ew", "passenger", "route_ew", "0.05"),
        ("flow_we", "passenger", "route_we", "0.05"),
        ("flow_ne", "passenger", "route_ne", "0.02"),
        ("flow_nw", "passenger", "route_nw", "0.02"),
        ("flow_se", "passenger", "route_se", "0.02"),
        ("flow_sw", "passenger", "route_sw", "0.02"),
        ("flow_en", "passenger", "route_en", "0.02"),
        ("flow_es", "passenger", "route_es", "0.02"),
        ("flow_wn", "passenger", "route_wn", "0.02"),
        ("flow_ws", "passenger", "route_ws", "0.02"),
        ("flow_truck_ns", "truck", "route_ns", "0.01"),
        ("flow_truck_ew", "truck", "route_ew", "0.01"),
    ]

    # Add all flows
    for i, (flow_id, vtype, route_id, prob) in enumerate(flow_definitions):
        flow = ET.SubElement(routes, "flow")
        flow.set("id", flow_id)
        flow.set("type", vtype)
        flow.set("route", route_id)
        flow.set("begin", "0")
        flow.set("end", "3600")
        flow.set("probability", prob)

    # Save the file
    with open(os.path.join("networks", "low_traffic.rou.xml"), "w") as f:
        f.write(prettify(routes))

    # Create a SUMO config file that references this route file
    create_sumo_config("low_traffic", "low_traffic.rou.xml")


def create_high_traffic_scenario():
    """Creates a high traffic volume scenario"""
    routes = ET.Element("routes")
    routes.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    routes.set(
        "xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd"
    )

    # Vehicle types
    vtype1 = ET.SubElement(routes, "vType")
    vtype1.set("id", "passenger")
    vtype1.set("length", "5.0")
    vtype1.set("minGap", "2.5")
    vtype1.set("maxSpeed", "15.0")
    vtype1.set("accel", "2.0")
    vtype1.set("decel", "4.5")
    vtype1.set("sigma", "0.5")

    vtype2 = ET.SubElement(routes, "vType")
    vtype2.set("id", "truck")
    vtype2.set("length", "7.5")
    vtype2.set("minGap", "3.0")
    vtype2.set("maxSpeed", "13.0")
    vtype2.set("accel", "1.5")
    vtype2.set("decel", "3.5")
    vtype2.set("sigma", "0.5")

    # Define all possible routes
    route_definitions = [
        ("route_ns", "north_to_center center_to_south"),
        ("route_ne", "north_to_center center_to_east"),
        ("route_nw", "north_to_center center_to_west"),
        ("route_sn", "south_to_center center_to_north"),
        ("route_se", "south_to_center center_to_east"),
        ("route_sw", "south_to_center center_to_west"),
        ("route_ew", "east_to_center center_to_west"),
        ("route_en", "east_to_center center_to_north"),
        ("route_es", "east_to_center center_to_south"),
        ("route_we", "west_to_center center_to_east"),
        ("route_wn", "west_to_center center_to_north"),
        ("route_ws", "west_to_center center_to_south"),
    ]

    # Add all routes
    for route_id, edges in route_definitions:
        route = ET.SubElement(routes, "route")
        route.set("id", route_id)
        route.set("edges", edges)

    # Define high traffic flows (probability >= 0.1)
    flow_definitions = [
        ("flow_ns", "passenger", "route_ns", "0.25"),
        ("flow_sn", "passenger", "route_sn", "0.25"),
        ("flow_ew", "passenger", "route_ew", "0.25"),
        ("flow_we", "passenger", "route_we", "0.25"),
        ("flow_ne", "passenger", "route_ne", "0.10"),
        ("flow_nw", "passenger", "route_nw", "0.10"),
        ("flow_se", "passenger", "route_se", "0.10"),
        ("flow_sw", "passenger", "route_sw", "0.10"),
        ("flow_en", "passenger", "route_en", "0.10"),
        ("flow_es", "passenger", "route_es", "0.10"),
        ("flow_wn", "passenger", "route_wn", "0.10"),
        ("flow_ws", "passenger", "route_ws", "0.10"),
        ("flow_truck_ns", "truck", "route_ns", "0.05"),
        ("flow_truck_ew", "truck", "route_ew", "0.05"),
    ]

    # Add all flows
    for i, (flow_id, vtype, route_id, prob) in enumerate(flow_definitions):
        flow = ET.SubElement(routes, "flow")
        flow.set("id", flow_id)
        flow.set("type", vtype)
        flow.set("route", route_id)
        flow.set("begin", "0")
        flow.set("end", "3600")
        flow.set("probability", prob)

    # Save the file
    with open(os.path.join("networks", "high_traffic.rou.xml"), "w") as f:
        f.write(prettify(routes))

    # Create a SUMO config file that references this route file
    create_sumo_config("high_traffic", "high_traffic.rou.xml")


def create_variable_demand_scenario():
    """Creates a variable demand scenario with changing traffic patterns"""
    routes = ET.Element("routes")
    routes.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    routes.set(
        "xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/routes_file.xsd"
    )

    # Vehicle types
    vtype1 = ET.SubElement(routes, "vType")
    vtype1.set("id", "passenger")
    vtype1.set("length", "5.0")
    vtype1.set("minGap", "2.5")
    vtype1.set("maxSpeed", "15.0")
    vtype1.set("accel", "2.0")
    vtype1.set("decel", "4.5")
    vtype1.set("sigma", "0.5")

    vtype2 = ET.SubElement(routes, "vType")
    vtype2.set("id", "truck")
    vtype2.set("length", "7.5")
    vtype2.set("minGap", "3.0")
    vtype2.set("maxSpeed", "13.0")
    vtype2.set("accel", "1.5")
    vtype2.set("decel", "3.5")
    vtype2.set("sigma", "0.5")

    # Define all possible routes
    route_definitions = [
        ("route_ns", "north_to_center center_to_south"),
        ("route_ne", "north_to_center center_to_east"),
        ("route_nw", "north_to_center center_to_west"),
        ("route_sn", "south_to_center center_to_north"),
        ("route_se", "south_to_center center_to_east"),
        ("route_sw", "south_to_center center_to_west"),
        ("route_ew", "east_to_center center_to_west"),
        ("route_en", "east_to_center center_to_north"),
        ("route_es", "east_to_center center_to_south"),
        ("route_we", "west_to_center center_to_east"),
        ("route_wn", "west_to_center center_to_north"),
        ("route_ws", "west_to_center center_to_south"),
    ]

    # Add all routes
    for route_id, edges in route_definitions:
        route = ET.SubElement(routes, "route")
        route.set("id", route_id)
        route.set("edges", edges)

    # Morning peak (high north->south, east->west)
    morning_flows = [
        ("flow_ns_morning", "passenger", "route_ns", "0", "1200", "0.30"),
        ("flow_sn_morning", "passenger", "route_sn", "0", "1200", "0.10"),
        ("flow_ew_morning", "passenger", "route_ew", "0", "1200", "0.25"),
        ("flow_we_morning", "passenger", "route_we", "0", "1200", "0.15"),
    ]

    # Midday (moderate traffic in all directions)
    midday_flows = [
        ("flow_ns_midday", "passenger", "route_ns", "1200", "2400", "0.15"),
        ("flow_sn_midday", "passenger", "route_sn", "1200", "2400", "0.15"),
        ("flow_ew_midday", "passenger", "route_ew", "1200", "2400", "0.15"),
        ("flow_we_midday", "passenger", "route_we", "1200", "2400", "0.15"),
    ]

    # Evening peak (high south->north, west->east)
    evening_flows = [
        ("flow_ns_evening", "passenger", "route_ns", "2400", "3600", "0.10"),
        ("flow_sn_evening", "passenger", "route_sn", "2400", "3600", "0.30"),
        ("flow_ew_evening", "passenger", "route_ew", "2400", "3600", "0.15"),
        ("flow_we_evening", "passenger", "route_we", "2400", "3600", "0.25"),
    ]

    # Add all time-varying flows
    for flows in [morning_flows, midday_flows, evening_flows]:
        for flow_id, vtype, route_id, begin, end, prob in flows:
            flow = ET.SubElement(routes, "flow")
            flow.set("id", flow_id)
            flow.set("type", vtype)
            flow.set("route", route_id)
            flow.set("begin", begin)
            flow.set("end", end)
            flow.set("probability", prob)

    # Add other consistent flows for all time periods
    other_flows = [
        ("flow_ne", "passenger", "route_ne", "0", "3600", "0.05"),
        ("flow_nw", "passenger", "route_nw", "0", "3600", "0.05"),
        ("flow_se", "passenger", "route_se", "0", "3600", "0.05"),
        ("flow_sw", "passenger", "route_sw", "0", "3600", "0.05"),
        ("flow_en", "passenger", "route_en", "0", "3600", "0.05"),
        ("flow_es", "passenger", "route_es", "0", "3600", "0.05"),
        ("flow_wn", "passenger", "route_wn", "0", "3600", "0.05"),
        ("flow_ws", "passenger", "route_ws", "0", "3600", "0.05"),
        ("flow_truck_ns", "truck", "route_ns", "0", "3600", "0.02"),
        ("flow_truck_ew", "truck", "route_ew", "0", "3600", "0.02"),
    ]

    for flow_id, vtype, route_id, begin, end, prob in other_flows:
        flow = ET.SubElement(routes, "flow")
        flow.set("id", flow_id)
        flow.set("type", vtype)
        flow.set("route", route_id)
        flow.set("begin", begin)
        flow.set("end", end)
        flow.set("probability", prob)

    # Save the file
    with open(os.path.join("networks", "variable_demand.rou.xml"), "w") as f:
        f.write(prettify(routes))

    # Create a SUMO config file that references this route file
    create_sumo_config("variable_demand", "variable_demand.rou.xml")


def create_sumo_config(scenario_name, route_file):
    """Create a SUMO configuration file for the given scenario"""
    config = ET.Element("configuration")
    config.set("xmlns:xsi", "http://www.w3.org/2001/XMLSchema-instance")
    config.set(
        "xsi:noNamespaceSchemaLocation", "http://sumo.dlr.de/xsd/sumoConfiguration.xsd"
    )

    input_section = ET.SubElement(config, "input")
    net_file = ET.SubElement(input_section, "net-file")
    net_file.set("value", "single_intersection.net.xml")
    route_files = ET.SubElement(input_section, "route-files")
    route_files.set("value", route_file)

    time_section = ET.SubElement(config, "time")
    begin = ET.SubElement(time_section, "begin")
    begin.set("value", "0")
    end = ET.SubElement(time_section, "end")
    end.set("value", "3600")
    step_length = ET.SubElement(time_section, "step-length")
    step_length.set("value", "1.0")

    processing = ET.SubElement(config, "processing")
    teleport = ET.SubElement(processing, "time-to-teleport")
    teleport.set("value", "-1")
    collision = ET.SubElement(processing, "collision.check-junctions")
    collision.set("value", "true")

    report = ET.SubElement(config, "report")
    verbose = ET.SubElement(report, "verbose")
    verbose.set("value", "true")
    step_log = ET.SubElement(report, "no-step-log")
    step_log.set("value", "true")

    # Save the file
    with open(os.path.join("networks", f"{scenario_name}.sumocfg"), "w") as f:
        f.write(prettify(config))


def main():
    # Make sure we're in the right directory
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    # Create different traffic scenarios
    print("Generating low traffic scenario...")
    create_low_traffic_scenario()

    print("Generating high traffic scenario...")
    create_high_traffic_scenario()

    print("Generating variable demand scenario...")
    create_variable_demand_scenario()

    print("Done! Traffic scenarios generated.")


if __name__ == "__main__":
    main()
