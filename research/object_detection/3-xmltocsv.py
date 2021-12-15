import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                     int(root.find('size')[0].text),
                     int(root.find('size')[1].text),
                     member[0].text,
                     int(member[4][0].text),
                     int(member[4][1].text),
                     int(member[4][2].text),
                     int(member[4][3].text)
                     )
            xml_list.append(value)
    column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    xml_df = pd.DataFrame(xml_list, columns=column_name)
    return xml_df



WORKS_NAME = 'complexity_analysis'

def main():

    for directory in ['train']:
        image_path = os.path.join(os.getcwd(), 'training_images/'+WORKS_NAME+'/train'.format(directory))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv('data/'+WORKS_NAME+'/train_labels.csv', index=None)
        print('Train Successfully converted xml to csv.')
        
    for directory in ['test']:
        image_path = os.path.join(os.getcwd(), 'training_images/'+WORKS_NAME+'/test'.format(directory))
        xml_df = xml_to_csv(image_path)
        xml_df.to_csv('data/'+WORKS_NAME+'/test_labels.csv', index=None)
        print('Test Successfully converted xml to csv.')

main()