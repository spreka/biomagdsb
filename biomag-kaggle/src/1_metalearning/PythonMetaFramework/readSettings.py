import xml.etree.ElementTree as ET

configFile = "D:\Ábel\SZBK\Projects\Kaggle\Abel\config.xml"

e = ET.parse(configFile).getroot()

tsDir = e.iterfind("trainingSet")

#runs only once because node trainingSet appears only once
for node in tsDir:
    L = list(node)
    for innerNode in L:
        if (innerNode.tag == "origDir"):
            origDir = innerNode.text
        if (innerNode.tag == "maskDir"):
            maskDir = innerNode.text;

