import os
from os import remove
from os.path import abspath, join

path =  abspath('cache')

for root, dirs, files in os.walk(path):
		for currentFile in files:
			print("processing file:",currentFile)
			ext = ('.bovw')
			if(currentFile.lower().endswith(ext)):
				print("Deleting:",currentFile)
				os.remove(os.path.join(root, currentFile))