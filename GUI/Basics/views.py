from django.shortcuts import render
import sklearn
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
# Create your views here.
def base(request):
    if request.method == "POST":
        data = request.POST
        SepalLengthCm = data.get('textsepallen')
        SepalWidthCm = data.get('textsepalwid')
        PetalLengthCm = data.get('textpetalle')
        PetalWidthCm = data.get('textpetalwid')
        if('buttonpredict' in request.POST):
            path = "/home/darpanyb/Documents/DARPAN/code project/github kt/Data/Iris.csv"
            data = pd.read_csv(path)
            print(data)

            inputs = data[['SepalLengthCm','SepalWidthCm','PetalLengthCm','PetalWidthCm']]
            output = data['Species']
            X_train, X_test, y_train, y_test = train_test_split(inputs, output, test_size=0.2, random_state=1)

            knn = KNeighborsClassifier(n_neighbors=13)
            knn.fit(X_train, y_train)

            result = knn.predict([[float(SepalLengthCm),float(SepalWidthCm),float(PetalLengthCm),float(PetalWidthCm)]])
        return render(request,'base.html',context={'result':result})
    return render(request,'base.html')
