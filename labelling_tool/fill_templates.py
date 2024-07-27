"""
# Fills the array of image names in index.js with the names of the images in the images folder
# Removes any existing names in the array
eg: 
var image_names = [
//////// FILL THE ARRAY WITH THE IMAGES YOU WANT TO LABEL ////////
  'images/0.jpg', 
  'images/60.jpg', 
  'images/120.jpg', 
  'images/180.jpg', 
  'images/240.jpg', 
  'images/300.jpg'
//////////////////////////////////////////////////////////////////
];
"""
import os
import click

def get_image_names(images_folder):
    image_names = []
    for image_name in os.listdir(images_folder):
        if image_name.endswith('.jpg'):
            image_names.append(f'images/{image_name}')
    # Sort the image names
    image_names.sort(key=lambda x: int(x.split('/')[-1].split('.')[0]))
    return image_names


def fill_image_names(images_folder, index_js_path):
    image_names = get_image_names(images_folder)
    with open(index_js_path, 'r') as f:
        lines = f.readlines()
    with open(index_js_path, 'w') as f:
        writing = True
        for line in lines:
            if '//////// FILL THE ARRAY WITH THE IMAGES YOU WANT TO LABEL ////////' in line:
                f.write(line)
                for image_name in image_names:
                    f.write(f"  '{image_name}',\n")
                writing = False
            else:
                if '//////////////////////////////////////////////////////////////////' in line:
                    writing = True
                if writing:
                    f.write(line)

def fill_lat_lng(lat: float, lng: float):
    """
    Replace the default latitude and longitude with the given values
    """
    index_js_path = 'index.js'
    with open(index_js_path, 'r') as f:
        lines = f.readlines()
    with open(index_js_path, 'w') as f:
        for line in lines:
            if 'const myLatlng = ' in line:
                f.write(f"    const myLatlng = {{lat: {lat}, lng: {lng}}};\n")
            else:
                f.write(line)

@click.command()
@click.option("--lat")
@click.option("--lng")
def cli(lat: str, lng: str):
    if len(lat) == 0:
        lat ="51.495878"
    if len(lng) == 0:
        lng ="-0.1422823"
    images_folder = 'images'
    index_js_path = 'index.js'
    fill_image_names(images_folder, index_js_path)
    print('Filled image names in index.js')
    fill_lat_lng(float(lat), float(lng))
    print(f'Filled latitude and longitude in index.js {lat}, {lng}')

if __name__ == '__main__':
    cli()
