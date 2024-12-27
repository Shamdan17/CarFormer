import os
import shutil
import xml.etree.ElementTree as ET

def split_routes(input_folder, output_folder):
    for dirpath, dirnames, filenames in os.walk(input_folder):
        # Exclude the root folder from the directory structure
        relative_dir = os.path.relpath(dirpath, input_folder)
        output_dir = os.path.join(output_folder, relative_dir)

        for dirname in dirnames:
            os.makedirs(os.path.join(output_dir, dirname), exist_ok=True)

        for filename in filenames:
            if filename.endswith(".xml"):
                input_path = os.path.join(dirpath, filename)
                filename_no_ext = os.path.splitext(filename)[0]

                tree = ET.parse(input_path)
                root = tree.getroot()

                for route in root.findall('route'):
                    route_id = route.get('id')
                    new_tree = ET.ElementTree(ET.Element('routes'))
                    new_root = new_tree.getroot()
                    new_root.append(route)

                    relative_path = os.path.relpath(input_path, input_folder)
                    output_file = os.path.join(output_folder, relative_path)
                    output_file = os.path.join(output_dir, f'{filename_no_ext}_{route_id}.xml')
                    
                    os.makedirs(os.path.dirname(output_file), exist_ok=True)
                    ET.indent(new_tree, '  ')
                    new_tree.write(output_file, encoding='utf-8', xml_declaration=True)

if __name__ == "__main__":
    input_folder = 'routes'
    output_folder = 'routes_splits'
    split_routes(input_folder, output_folder)

