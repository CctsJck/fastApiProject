from pdfminer.pdfdocument import PDFDocument, PDFNoOutlines
from pdfminer.pdfparser import PDFParser
from pdfminer.pdfinterp import PDFResourceManager, PDFPageInterpreter
from pdfminer.converter import PDFPageAggregator
from pdfminer.layout import LTPage, LTChar, LTAnno, LAParams, LTTextBox, LTTextLine
import numpy as np
import pandas as pd
from pprint import pprint
from pdfminer.pdfpage import PDFPage
from decimal import Decimal


class PDFPageDetailedAggregator(PDFPageAggregator):
    def __init__(self, rsrcmgr, pageno=1, laparams=None):
        PDFPageAggregator.__init__(self, rsrcmgr, pageno=pageno, laparams=laparams)
        self.rows = []
        self.page_number = 0
    def receive_layout(self, ltpage):
        def render(item, page_number):
            if isinstance(item, LTPage) or isinstance(item, LTTextBox):
                for child in item:
                    render(child, page_number)
            elif isinstance(item, LTTextLine):
                child_str = ''
                for child in item:
                    if isinstance(child, (LTChar, LTAnno)):
                        child_str += child.get_text()
                child_str = ' '.join(child_str.split()).strip()
                if child_str:
                    row = [page_number, item.bbox[0], item.bbox[1], item.bbox[2], item.bbox[3], child_str] # bbox == (x1, y1, x2, y2)
                    self.rows.append(row)
                for child in item:
                    render(child, page_number)
            return
        render(ltpage, self.page_number)
        self.page_number += 1
        self.rows = sorted(self.rows, key = lambda x: (x[0], -x[2]))
        self.result = ltpage



fp = open('./Profile.pdf', 'rb')
parser = PDFParser(fp)
doc = PDFDocument(parser)

rsrcmgr = PDFResourceManager()
laparams = LAParams()
device = PDFPageDetailedAggregator(rsrcmgr, laparams=laparams)
interpreter = PDFPageInterpreter(rsrcmgr, device)

for page in PDFPage.create_pages(doc):
    interpreter.process_page(page)
    # receive the LTPage object for this page
    device.get_result()

my_array = device.rows

df = pd.DataFrame.from_records(my_array)
df.columns = ['pn','x0','y0','x1','y1','line']
df['y_dif'] = round(df['y1']-df['y0'])

grouped_y0 = df.groupby(['pn'])['y0'].max().reset_index().sort_values('pn', ascending=True)
grouped_y1 = df.groupby(['pn'])['y1'].max().reset_index().sort_values('pn', ascending=True)

grouped_y0['y0_'] = grouped_y0['y0'].shift(-1)
grouped_y1['y1_'] = grouped_y1['y1'].shift(-1)

grouped_y0.fillna(0, inplace=True)
grouped_y1.fillna(0, inplace=True)

grouped_y0['cum_y0'] = grouped_y0.loc[::-1, 'y0_'].cumsum()[::-1]
grouped_y1['cum_y1'] = grouped_y1.loc[::-1, 'y1_'].cumsum()[::-1]

grouped_y0 = grouped_y0[['pn','cum_y0']]
grouped_y1 = grouped_y1[['pn','cum_y1']]

df = df.merge(grouped_y0, on='pn')
df = df.merge(grouped_y1, on='pn')

df['y0'] = df['y0'] + df['cum_y0']
df['y1'] = df['y1'] + df['cum_y1']

df.drop(['cum_y0', 'cum_y1'], axis=1, inplace=True)

indent_cat = {}
pct = 0.001
for index, row in df.iterrows():
    feature = row['x0']
    bin_found = False
    for bin_key, bin_arr in indent_cat.items():
        if feature >= (bin_key * (1.0 - pct)) and (bin_key * (1.0 + pct)) >= feature:
            bin_found = True
            bin_arr.append(index)
    if not bin_found:
        indent_cat[feature] = [index]


def labeling(x, cat_):
    for key, val in cat_.items():
        if x in val:
            return list(sorted(cat_)).index(key)


df['indent_label'] = df.index.map(lambda x: labeling(x, indent_cat))

df_1 = df[df['indent_label'] == 1].copy()
df_0 = df[df['indent_label'] == 0].copy()

arr = []


def title(row):
    if row.y_dif >= 22:
        arr.append(row.line)

    return arr[-1]


df_1['Title'] = df_1.apply(lambda row: title(row), axis=1)

df_1

df_ex = df_1[df_1['Title'] == 'Experience'].copy()
df_ed = df_1[df_1['Title'] == 'Education'].copy()

import re

pattern = r'([a-zA-Z]+\s\d{4})\s-\s(?:([a-zA-Z]+\s\d{4})|(\w+))'
regex = re.compile(pattern, re.IGNORECASE)

i = - 1

arr = []

for index, row in df_ex.iterrows():

    if row['y_dif'] == 17:

        arr.append({})
        arr[-1]['company'] = row['line']

    elif row['y_dif'] == 16:
        if 'experience' not in arr[-1]:
            arr[-1]['experience'] = []

        arr[-1]['experience'].append({})
        arr[-1]['experience'][-1]['position'] = row['line']

    elif row['y_dif'] == 15 and regex.findall(row['line']):
        if 'experience' in arr[-1]:
            t = arr[-1]['experience'][-1]['date_period'] if ('date_period' in arr[-1]['experience'][-1]) else []
            arr[-1]['experience'][-1]['date_period'] = t + regex.findall(row['line'])

    elif row['y_dif'] == 15:
        if 'experience' in arr[-1]:
            t = arr[-1]['experience'][-1]['meta'] if ('meta' in arr[-1]['experience'][-1]) else ''
            arr[-1]['experience'][-1]['meta'] = t + ' ' + row['line']

arr

print("Este es el array con el json de experiencia:")
print(arr)

i = - 1

arr_ed = []

for index, row in df_ed.iterrows():

    if row['y_dif'] == 17:

        arr_ed.append({})
        arr_ed[-1]['institute'] = row['line']

    elif row['y_dif'] == 15:
        t = arr_ed[-1]['degree'] if ('degree' in arr_ed[-1]) else ''
        arr_ed[-1]['degree'] = t + ' ' + row['line']
arr_ed


print("este es el array de educacion: ")
print(arr_ed)

arr = []
def title(row):
    if row.y_dif >= 18:
        arr.append(row.line)
    return arr[-1]

df_0['Title'] = df_0.apply(lambda row: title(row), axis = 1)

df_0

df_co = df_0[df_0['Title'] == 'Contact'].copy()
df_sk = df_0[df_0['Title'] == 'Top Skills'].copy()
df_la = df_0[df_0['Title'] == 'Languages'].copy()
df_cert = df_0[df_0['Title'] == 'Certifications'].copy()

i = - 1

arr_co = []

for index, row in df_co.iterrows():

    if row['y_dif'] == 18:
        arr_co.append({})

    if row['y_dif'] == 15:
        t = arr_co[-1]['contact'] if ('contact' in arr_co[-1]) else []
        arr_co[-1]['contact'] = t + [row['line']]
i = - 1

print("Este es el array con el json de contacto:")
print(arr_co)

arr_sk = []

for index, row in df_sk.iterrows():

    if row['y_dif'] == 18:
        arr_sk.append({})

    if row['y_dif'] == 15:
        t = arr_sk[-1]['skills'] if ('skills' in arr_sk[-1]) else []
        arr_sk[-1]['skills'] = t + [row['line']]
i = - 1

print("Este es el array con el json de top skills:")
print(arr_sk)

arr_la = []

for index, row in df_la.iterrows():

    if row['y_dif'] == 18:
        arr_la.append({})

    if row['y_dif'] == 15:
        t = arr_la[-1]['languages'] if ('languages' in arr_la[-1]) else []
        arr_la[-1]['languages'] = t + [row['line']]
i = - 1

print("Este es el array con el json de idiomas:")
print(arr_la)

arr_cert = []

for index, row in df_cert.iterrows():

    if row['y_dif'] == 18:
        arr_cert.append({})

    if row['y_dif'] == 15:
        t = arr_cert[-1]['certifications'] if ('certifications' in arr_cert[-1]) else []
        arr_cert[-1]['certifications'] = t + [row['line']]

arr_cert

print("Este es el array con el json de certificaciones:")
print(arr_cert[0].get('certifications'))

