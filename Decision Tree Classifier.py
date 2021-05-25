import numpy as np
import pandas as pd
from sklearn import tree
 
input_file = "/content/Book1.csv"
df = pd.read_csv(input_file, header=0)

d = {'Y': 1, 'N':0}
df['Joined Teams'] = df['Joined Teams'].map(d) 
df['Joined ERP'] = df['Joined ERP'].map(d)
df['Name Box Ticked'] = df['Name Box Ticked'].map(d)
df['Attendance Marked'] = df['Attendance Marked'].map(d)

df.head()
features = list(df.columns[:3])
features

y = df['Attendance Marked']
x = df[features]
clf = tree.DecisionTreeClassifier()
clf = clf.fit(x,y)

from IPython.display import Image
from sklearn.externals.six import StringIO
import pydot

dot_data=StringIO()
tree.export_graphviz(clf, out_file=dot_data, feature_names=features)
graph = pydot.graph_from_dot_data(dot_data.getvalue())
Image(graph[0].create_png())