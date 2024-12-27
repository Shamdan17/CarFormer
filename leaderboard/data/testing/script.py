import xml.etree.ElementTree as ET

def split_routes(xml_file):
    tree = ET.parse(xml_file)
    root = tree.getroot()

    for route in root.findall('route'):
        route_id = route.get('id')
        new_tree = ET.ElementTree(ET.Element('routes'))
        new_root = new_tree.getroot()
        new_root.append(route)
        
        new_xml_file = f'route_{route_id}.xml'
        new_tree.write(new_xml_file, encoding='utf-8', xml_declaration=True)

if __name__ == "__main__":
    xml_file = 'routes.xml'
    split_routes(xml_file)

