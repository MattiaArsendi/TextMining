import PyPDF2
from PyPDF2 import PdfFileWriter
from PyPDF2 import PdfFileReader
from wordcloud import WordCloud
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
from tika import parser

#text = textract.process(r'C:/Users/Francesco/Desktop/LIBRI DA LEGGE FRANCE/fake news detection.pdf', method='pdfminer')
#stdout, stderr = popen.communicate()
file=open(r'C:/Users/Francesco/Desktop/LIBRI DA LEGGE FRANCE/the element of statistical learning.pdf', 'rb') # QUESTO È PERSONALE
fileReader = PyPDF2.PdfFileReader(file)
print(fileReader.numPages)

pageData = ''
for page in fileReader.pages:
    pageData += page.extractText()
    #print(pageData)

raw = parser.from_file('sample.pdf')
print(raw['content'])

#pageObj =fileReader.getPage(100)
#print(pageObj.extractText())
#fileReader.close()

cloud = WordCloud().generate(pageData)#ancora non va bene
# MA LAVORANDO CI AVVICINIAMO !! NIENTE VIENTE DA NIENTE.

plt.imshow(cloud)
plt.axis('off')
plt.show() #ciaooo













#questa parte che viene non funziona





def split_pdf_to_two(filename,page_number):
    pdf_reader = PdfFileReader(open(filename, "rb"))
    try:
        assert page_number < pdf_reader.numPages
        pdf_writer1 = PdfFileWriter()
        pdf_writer2 = PdfFileWriter()

        for page in range(page_number):
            pdf_writer1.addPage(pdf_reader.getPage(page))

        for page in range(page_number,pdf_reader.getNumPages()):
            pdf_writer2.addPage(pdf_reader.getPage(page))

        with open("part1.pdf", 'wb') as file1:
            pdf_writer1.write(file1)

        with open("part2.pdf", 'wb') as file2:
            pdf_writer2.write(file2)
            
        print(pdf_writer2.numPages)

    except AssertionError as e:
        print("Error: The PDF you are cutting has less pages than you want to cut!")
        
split_pdf_to_two(r'C:/Users/Francesco/Desktop/LIBRI DA LEGGE FRANCE/the element of statistical learning.pdf',100) 