import streamlit as st
import numpy as np
from sklearn import datasets
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

st.title("streamlit example")

st.write("""
# Explore diffrent classifier
which one is best
""")
# select datasets................
datasets_name = st.sidebar.selectbox("select Dataset",("Iris","Breast cancer","Wine dataset"))
st.write("Dataset - ",datasets_name)
classifier_name = st.sidebar.selectbox("select classifier",("KNN","SVM","Random forest","linear Regrassion"))
st.write("Classifier name -",classifier_name )

def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()
    x = data.data
    y = data.target
    return x,y

x, y = get_dataset(datasets_name)
st.write("shape of dataset",x.shape)
st.write("number of classes",len(np.unique(y)))

def add_parameter_ui(clf_name):
    params = dict()
    if clf_name == "KNN":
        k = st.sidebar.slider("k", 1, 15)
        params["k"] = k
    elif clf_name == "SVM":
        C = st.sidebar.slider("c",0.01,10.0)  
        params["c"] = C
    else:
        max_depth = st.sidebar.slider("max_depth",2, 15)
        n_estimators = st.sidebar.slider("n_estimators",1,100)
        params["max_depth"] = max_depth
        params["n_estimators"] = n_estimators
        
    return params

params = add_parameter_ui(classifier_name)

def get_classifier(clf_name,params):

    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors = params["k"])
      # k = st.sidebar.slider("k", 1, 15)
      # params["k"] = k
    elif clf_name == "SVM":
        clf = SVC(C = params["C"] )
          # params["C"] = C
    else:
        clf = RandomForestClassifier(n_estimators = params["n_estimators"],
                     max_depth= params["max_depth"],random_state = 1234)
    return clf


clf = get_classifier(classifier_name,params)

X_tain, X_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state = 1234)

clf.fit(X_tain, y_train)
y_predit = clf.predict(X_test)

acc = accuracy_score(y_test,y_predit)
st.write(f"classifier = {classifier_name}")
st.write(f"accuracy = {acc}")


# plot..............

pca = PCA(2)
X_projected =pca.fit_transform(x)

x1 = X_projected[:,0]
x2 = X_projected[:,1]

fig = plt.figure()
plt.scatter(x1,x2, c=y, alpha=0.8,cmap="viridis")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.colorbar()

st.pyplot(fig)
